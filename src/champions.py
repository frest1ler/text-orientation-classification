"""Per-model champion registry backed by atomic filesystem updates."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SOURCES = (
    "src/synthetic.py",
    "src/text_corpus.py",
    "assets/fonts/manifest.json",
)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validation_protocol(
    config: dict[str, Any], runtime: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Build a validation fingerprint that blocks incomparable promotion."""
    source_hashes = {
        relative: _file_sha256(PROJECT_ROOT / relative) for relative in PROTOCOL_SOURCES
    }
    protocol = {
        "seed": config["experiment"]["seed"],
        "validation_base_samples": runtime["validation_base_samples"],
        "synthetic": config["synthetic"],
        "source_sha256": source_hashes,
    }
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest(), protocol


def _ranking_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    symmetric = metrics["symmetric"]
    return (
        float(symmetric["brier_score"]),
        float(symmetric["log_loss"]),
        -float(symmetric["roc_auc"]),
        -float(symmetric["accuracy"]),
    )


def _safe_model_name(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("._")
    if not safe:
        raise ValueError("model name does not contain safe filename characters")
    return safe


def _load_candidate(run_dir: Path) -> dict[str, Any]:
    required = (
        "best.pt",
        "calibration.json",
        "config.json",
        "runtime.json",
        "history.json",
        "environment.json",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Run directory is missing required files: {missing}")
    config = _read_json(run_dir / "config.json")
    runtime = _read_json(run_dir / "runtime.json")
    history = _read_json(run_dir / "history.json")
    environment = _read_json(run_dir / "environment.json")
    calibration = _read_json(run_dir / "calibration.json")
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    if not history:
        raise ValueError("history.json is empty")
    best_record = min(history, key=lambda item: item["symmetric"]["brier_score"])
    if checkpoint["epoch"] != best_record["epoch"]:
        raise ValueError("checkpoint epoch does not match best history epoch")
    if checkpoint["metrics"] != {
        key: value
        for key, value in best_record.items()
        if key not in {"epoch", "phase", "loss", "classification_loss", "symmetry_loss"}
    }:
        raise ValueError("checkpoint metrics do not match best history metrics")
    if calibration.get("checkpoint_epoch") != checkpoint["epoch"]:
        raise ValueError("calibration checkpoint epoch does not match best checkpoint")
    if calibration.get("prediction_mode") != "symmetric":
        raise ValueError("calibration must be fitted on symmetric predictions")
    final_calibrator = calibration.get("final_calibrator")
    if not isinstance(final_calibrator, dict) or final_calibrator.get("method") not in {
        "uncalibrated",
        "temperature",
        "platt",
    }:
        raise ValueError("calibration method is unsupported")
    if not all(
        isinstance(final_calibrator.get(key), (int, float))
        and math.isfinite(final_calibrator[key])
        for key in ("slope", "intercept")
    ):
        raise ValueError("calibration parameters must be finite")
    protocol_hash, protocol = validation_protocol(config, runtime)
    return {
        "config": config,
        "runtime": runtime,
        "environment": environment,
        "checkpoint": checkpoint,
        "calibration": calibration,
        "protocol_hash": protocol_hash,
        "protocol": protocol,
    }


def _refresh_leaderboard(registry_dir: Path) -> dict[str, Any]:
    champions_root = registry_dir / "champions"
    models: dict[str, Any] = {}
    if champions_root.is_dir():
        for manifest_path in sorted(champions_root.glob("*/champion.json")):
            manifest = _read_json(manifest_path)
            models[manifest["model"]] = {
                "checkpoint": manifest["checkpoint"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "calibration": manifest["calibration"],
                "calibration_sha256": manifest["calibration_sha256"],
                "metrics": manifest["metrics"],
                "epoch": manifest["epoch"],
                "validation_protocol_sha256": manifest["validation_protocol_sha256"],
            }
    leaderboard = {"models": models}
    _write_json_atomic(registry_dir / "leaderboard.json", leaderboard)
    return leaderboard


def promote_champion(run_dir: str | Path, registry_dir: str | Path) -> dict[str, Any]:
    """Promote a run when it beats the comparable champion of the same model."""
    run_dir = Path(run_dir)
    registry_dir = Path(registry_dir)
    candidate = _load_candidate(run_dir)
    config = candidate["config"]
    model_name = _safe_model_name(config["model"]["name"])
    metrics = candidate["checkpoint"]["metrics"]
    accuracy = float(metrics["symmetric"]["accuracy"])
    checkpoint_name = f"{model_name}_{accuracy:.6f}.pt"
    model_dir = registry_dir / "champions" / model_name
    manifest_path = model_dir / "champion.json"
    existing = _read_json(manifest_path) if manifest_path.is_file() else None
    candidate_checkpoint_hash = _file_sha256(run_dir / "best.pt")

    if existing is not None and (
        existing["validation_protocol_sha256"] != candidate["protocol_hash"]
    ):
        return {
            "status": "incompatible_validation_protocol",
            "model": model_name,
            "candidate_protocol": candidate["protocol_hash"],
            "champion_protocol": existing["validation_protocol_sha256"],
        }
    existing_complete = False
    if existing is not None and all(
        key in existing
        for key in ("calibration", "calibration_sha256", "checkpoint_sha256")
    ):
        existing_checkpoint = model_dir / existing["checkpoint"]
        existing_calibration = model_dir / existing["calibration"]
        existing_complete = (
            existing_checkpoint.is_file()
            and existing_calibration.is_file()
            and _file_sha256(existing_checkpoint) == existing["checkpoint_sha256"]
            and _file_sha256(existing_calibration) == existing["calibration_sha256"]
        )
    if existing is not None and not existing_complete and (
        existing.get("checkpoint_sha256") != candidate_checkpoint_hash
    ):
        raise ValueError(
            "existing champion bundle is incomplete and cannot be repaired from a different run"
        )
    if existing_complete and _ranking_key(metrics) >= _ranking_key(existing["metrics"]):
        return {
            "status": "kept_existing",
            "model": model_name,
            "champion": existing["checkpoint"],
            "candidate_brier": metrics["symmetric"]["brier_score"],
            "champion_brier": existing["metrics"]["symmetric"]["brier_score"],
        }

    model_dir.mkdir(parents=True, exist_ok=True)
    destination = model_dir / checkpoint_name
    temporary = model_dir / f".{checkpoint_name}.tmp"
    shutil.copy2(run_dir / "best.pt", temporary)
    os.replace(temporary, destination)
    calibration_name = "calibration.json"
    calibration_destination = model_dir / calibration_name
    calibration_temporary = model_dir / f".{calibration_name}.tmp"
    shutil.copy2(run_dir / calibration_name, calibration_temporary)
    os.replace(calibration_temporary, calibration_destination)
    manifest = {
        "model": model_name,
        "checkpoint": checkpoint_name,
        "checkpoint_sha256": candidate_checkpoint_hash,
        "calibration": calibration_name,
        "calibration_sha256": _file_sha256(calibration_destination),
        "epoch": candidate["checkpoint"]["epoch"],
        "metrics": metrics,
        "validation_protocol_sha256": candidate["protocol_hash"],
        "validation_protocol": candidate["protocol"],
        "runtime": candidate["runtime"],
        "environment": candidate["environment"],
    }
    _write_json_atomic(manifest_path, manifest)
    if existing is not None and existing["checkpoint"] != checkpoint_name:
        old_checkpoint = model_dir / existing["checkpoint"]
        if old_checkpoint.is_file():
            old_checkpoint.unlink()
    leaderboard = _refresh_leaderboard(registry_dir)
    return {
        "status": "promoted",
        "model": model_name,
        "checkpoint": str(destination),
        "metrics": metrics,
        "leaderboard": leaderboard,
    }
