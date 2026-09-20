"""Validation search and registry contracts for probability ensembles."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from src.metrics import binary_metrics
from src.registry import ChampionBundle, file_sha256, select_champion


ENSEMBLE_VERSION = 1


@dataclass(frozen=True)
class EnsembleCandidate:
    selector: str
    bundle: ChampionBundle
    targets: np.ndarray
    sample_ids: np.ndarray
    probabilities: np.ndarray
    prediction_source: str


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_validation_candidate(
    registry_dir: str | Path, selector: str, run_dir: str | Path
) -> EnsembleCandidate:
    """Load protocol-checked OOF calibrated predictions for one champion."""
    bundle = select_champion(registry_dir, selector)
    run_dir = Path(run_dir)
    predictions_path = run_dir / "validation_predictions.npz"
    calibration_path = run_dir / "calibration.json"
    checkpoint_path = run_dir / "best.pt"
    for path in (predictions_path, calibration_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(f"ensemble candidate file is missing: {path}")
    if file_sha256(checkpoint_path) != bundle.manifest["checkpoint_sha256"]:
        raise ValueError(f"candidate checkpoint does not match registry selector '{selector}'")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    method = calibration.get("selected_method")
    prediction_key = "oof_uncalibrated" if method == "uncalibrated" else f"oof_{method}"
    with np.load(predictions_path, allow_pickle=False) as values:
        if "targets" not in values or prediction_key not in values:
            raise ValueError(
                f"validation predictions require targets and {prediction_key}"
            )
        targets = np.asarray(values["targets"], dtype=np.int64).reshape(-1)
        probabilities = np.asarray(values[prediction_key], dtype=np.float64).reshape(-1)
        sample_ids = (
            np.asarray(values["sample_ids"]).astype(str).reshape(-1)
            if "sample_ids" in values
            else np.arange(targets.size).astype(str)
        )
    binary_metrics(targets, probabilities)
    if sample_ids.shape != targets.shape or np.unique(sample_ids).size != sample_ids.size:
        raise ValueError("validation sample IDs must be unique and aligned")
    return EnsembleCandidate(
        selector, bundle, targets, sample_ids, probabilities, prediction_key
    )


def validate_candidates(candidates: Sequence[EnsembleCandidate]) -> None:
    if len(candidates) < 2:
        raise ValueError("an ensemble requires at least two candidates")
    selectors = [candidate.selector for candidate in candidates]
    if len(set(selectors)) != len(selectors):
        raise ValueError("ensemble candidate selectors must be unique")
    protocols = {
        candidate.bundle.manifest["validation_protocol_sha256"]
        for candidate in candidates
    }
    if len(protocols) != 1:
        raise ValueError("ensemble candidates use incompatible validation protocols")
    reference = candidates[0]
    for candidate in candidates[1:]:
        if not np.array_equal(reference.sample_ids, candidate.sample_ids):
            raise ValueError("ensemble validation sample order differs")
        if not np.array_equal(reference.targets, candidate.targets):
            raise ValueError("ensemble validation targets differ")


def _weight_grid(count: int, step: float) -> Iterable[tuple[float, ...]]:
    units = round(1.0 / step)
    if count < 2 or not math.isclose(units * step, 1.0, abs_tol=1e-9):
        raise ValueError("weight-step must divide 1 exactly")

    def compositions(remaining: int, parts: int, prefix: tuple[int, ...]):
        if parts == 1:
            yield (*prefix, remaining)
            return
        for value in range(remaining + 1):
            yield from compositions(remaining - value, parts - 1, (*prefix, value))

    for values in compositions(units, count, ()):
        if all(value > 0 for value in values):
            yield tuple(value / units for value in values)


def search_ensemble(
    candidates: Sequence[EnsembleCandidate], weight_step: float
) -> dict[str, Any]:
    """Deterministically minimize validation Brier over a simplex grid."""
    validate_candidates(candidates)
    matrix = np.stack([candidate.probabilities for candidate in candidates])
    best: dict[str, Any] | None = None
    evaluated = 0
    for weights in _weight_grid(len(candidates), weight_step):
        probabilities = np.average(matrix, axis=0, weights=np.asarray(weights))
        metrics = binary_metrics(candidates[0].targets, probabilities)
        record = {"weights": list(weights), "metrics": metrics}
        key = (
            metrics["brier_score"],
            metrics["log_loss"],
            -metrics["roc_auc"],
            tuple(weights),
        )
        if best is None or key < best["_key"]:
            best = {**record, "_key": key}
        evaluated += 1
    if best is None:
        raise ValueError("weight grid contains no strictly positive combinations")
    best.pop("_key")
    return {"evaluated": evaluated, **best}


def build_ensemble_manifest(
    candidates: Sequence[EnsembleCandidate], search: dict[str, Any]
) -> dict[str, Any]:
    validate_candidates(candidates)
    weights = search["weights"]
    if len(weights) != len(candidates):
        raise ValueError("weight count does not match candidates")
    components = []
    component_briers = []
    for candidate, weight in zip(candidates, weights, strict=True):
        candidate_metrics = binary_metrics(candidate.targets, candidate.probabilities)
        component_briers.append(candidate_metrics["brier_score"])
        components.append(
            {
                "selector": candidate.selector,
                "model": candidate.bundle.manifest["model"],
                "weight": float(weight),
                "checkpoint_sha256": candidate.bundle.manifest["checkpoint_sha256"],
                "calibration_sha256": candidate.bundle.manifest["calibration_sha256"],
                "prediction_source": candidate.prediction_source,
                "validation_metrics": candidate_metrics,
            }
        )
    payload = {
        "version": ENSEMBLE_VERSION,
        "name": "best_ensemble",
        "validation_protocol_sha256": candidates[0].bundle.manifest[
            "validation_protocol_sha256"
        ],
        "components": components,
        "validation_metrics": search["metrics"],
        "improves_best_component": float(search["metrics"]["brier_score"])
        < min(component_briers),
        "evaluated_weight_combinations": search["evaluated"],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "ensemble_sha256": hashlib.sha256(canonical).hexdigest()}


def promote_ensemble(manifest: dict[str, Any], registry_dir: str | Path) -> dict[str, Any]:
    """Atomically keep the lower-Brier isolated ensemble champion."""
    root = Path(registry_dir) / "ensembles"
    if not manifest.get("improves_best_component"):
        return {"status": "not_better_than_component", "manifest": manifest}
    destination = root / "champions" / "best_ensemble" / "ensemble.json"
    existing = json.loads(destination.read_text(encoding="utf-8")) if destination.is_file() else None
    candidate_brier = float(manifest["validation_metrics"]["brier_score"])
    if existing is not None:
        if existing["validation_protocol_sha256"] != manifest["validation_protocol_sha256"]:
            raise ValueError("ensemble candidate protocol differs from existing champion")
        if candidate_brier >= float(existing["validation_metrics"]["brier_score"]):
            return {"status": "kept_existing", "manifest": existing}
    _atomic_json(destination, manifest)
    leaderboard = {
        "ensembles": {
            "best_ensemble": {
                "manifest": "champions/best_ensemble/ensemble.json",
                "ensemble_sha256": manifest["ensemble_sha256"],
                "validation_protocol_sha256": manifest[
                    "validation_protocol_sha256"
                ],
                "validation_metrics": manifest["validation_metrics"],
            }
        }
    }
    _atomic_json(root / "leaderboard.json", leaderboard)
    return {"status": "promoted", "manifest": manifest, "leaderboard": leaderboard}


def load_ensemble(registry_dir: str | Path, name: str = "best_ensemble") -> dict[str, Any]:
    path = Path(registry_dir) / "ensembles" / "champions" / name / "ensemble.json"
    if not path.is_file():
        raise FileNotFoundError(f"ensemble manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != ENSEMBLE_VERSION or manifest.get("name") != name:
        raise ValueError("ensemble manifest is incompatible")
    stored_hash = manifest.get("ensemble_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "ensemble_sha256"}
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    if stored_hash != hashlib.sha256(canonical).hexdigest():
        raise ValueError("ensemble manifest SHA-256 mismatch")
    components = manifest.get("components")
    if not isinstance(components, list) or len(components) < 2:
        raise ValueError("ensemble manifest requires at least two components")
    total = sum(float(component.get("weight", -1)) for component in components)
    if any(float(component.get("weight", -1)) <= 0 for component in components):
        raise ValueError("ensemble weights must be positive")
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ValueError("ensemble weights must sum to one")
    protocol = manifest.get("validation_protocol_sha256")
    for component in components:
        bundle = select_champion(registry_dir, component["selector"])
        if bundle.manifest["checkpoint_sha256"] != component["checkpoint_sha256"]:
            raise ValueError("ensemble component checkpoint changed")
        if bundle.manifest["calibration_sha256"] != component["calibration_sha256"]:
            raise ValueError("ensemble component calibration changed")
        if bundle.manifest["validation_protocol_sha256"] != protocol:
            raise ValueError("ensemble component validation protocol changed")
    return manifest


def combine_prediction_rows(
    component_rows: Sequence[Sequence[dict[str, Any]]], weights: Sequence[float]
) -> list[dict[str, float | str]]:
    """Combine aligned inference rows without loading models concurrently."""
    if len(component_rows) < 2 or len(component_rows) != len(weights):
        raise ValueError("component rows and weights are inconsistent")
    if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
        raise ValueError("ensemble weights must sum to one")
    size = len(component_rows[0])
    if any(len(rows) != size for rows in component_rows):
        raise ValueError("ensemble component prediction lengths differ")
    output = []
    for index in range(size):
        identifiers = [str(rows[index]["image_id"]) for rows in component_rows]
        if len(set(identifiers)) != 1:
            raise ValueError("ensemble component image order differs")
        values = {}
        for key in ("p_direct", "p_rotated", "p_symmetric", "p_final"):
            value = sum(
                float(weight) * float(rows[index][key])
                for rows, weight in zip(component_rows, weights, strict=True)
            )
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("ensemble produced an invalid probability")
            values[key] = value
        output.append(
            {
                "image_id": identifiers[0],
                **values,
                "symmetry_error": abs(values["p_direct"] + values["p_rotated"] - 1),
            }
        )
    return output
