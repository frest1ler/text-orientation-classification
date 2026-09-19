"""Validated model-registry selection for training and inference clients."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required registry file is missing: {path}")
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Registry JSON must contain an object: {path}")
    return value


@dataclass(frozen=True)
class ChampionBundle:
    model: str
    directory: Path
    checkpoint_path: Path
    manifest: dict[str, Any]
    calibration: dict[str, Any]


def _leaderboard_record(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint": manifest["checkpoint"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "calibration": manifest["calibration"],
        "calibration_sha256": manifest["calibration_sha256"],
        "metrics": manifest["metrics"],
        "epoch": manifest["epoch"],
        "validation_protocol_sha256": manifest["validation_protocol_sha256"],
    }


def _validate_bundle(directory: Path) -> ChampionBundle:
    manifest = _read_json(directory / "champion.json")
    required = {
        "model",
        "checkpoint",
        "checkpoint_sha256",
        "calibration",
        "calibration_sha256",
        "epoch",
        "metrics",
        "validation_protocol_sha256",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"champion manifest is missing required fields: {missing}")
    model = manifest.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("champion manifest has an invalid model name")
    checkpoint_path = directory / manifest["checkpoint"]
    calibration_path = directory / manifest["calibration"]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Champion checkpoint is missing: {checkpoint_path}")
    if file_sha256(checkpoint_path) != manifest["checkpoint_sha256"]:
        raise ValueError("champion checkpoint SHA-256 mismatch")
    if file_sha256(calibration_path) != manifest["calibration_sha256"]:
        raise ValueError("champion calibration SHA-256 mismatch")
    calibration = _read_json(calibration_path)
    if calibration.get("checkpoint_epoch") != manifest["epoch"]:
        raise ValueError("calibration checkpoint epoch does not match champion")
    if calibration.get("prediction_mode") != "symmetric":
        raise ValueError("champion calibration must target symmetric predictions")
    final = calibration.get("final_calibrator")
    if not isinstance(final, dict) or final.get("method") not in {
        "uncalibrated",
        "temperature",
        "platt",
    }:
        raise ValueError("champion calibration method is unsupported")
    parameters = (final.get("slope"), final.get("intercept"))
    if not all(isinstance(value, (int, float)) and np.isfinite(value) for value in parameters):
        raise ValueError("champion calibration parameters must be finite")
    return ChampionBundle(model, directory, checkpoint_path, manifest, calibration)


def select_champion(registry_or_bundle: str | Path, model: str = "best") -> ChampionBundle:
    """Select and validate the best registry model or a specific champion."""
    root = Path(registry_or_bundle)
    if (root / "champion.json").is_file():
        bundle = _validate_bundle(root)
        if model not in {"best", bundle.model}:
            raise ValueError(
                f"Uploaded bundle contains '{bundle.model}', not requested '{model}'"
            )
        return bundle

    leaderboard = _read_json(root / "leaderboard.json")
    models = leaderboard.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("leaderboard does not contain any model champions")
    if model == "best":
        selected = min(
            models,
            key=lambda name: (
                float(models[name]["metrics"]["symmetric"]["brier_score"]),
                float(models[name]["metrics"]["symmetric"]["log_loss"]),
                -float(models[name]["metrics"]["symmetric"]["roc_auc"]),
                -float(models[name]["metrics"]["symmetric"]["accuracy"]),
            ),
        )
    else:
        if model not in models:
            raise ValueError(f"Model '{model}' is absent from the leaderboard")
        selected = model
    bundle = _validate_bundle(root / "champions" / selected)
    if bundle.model != selected:
        raise ValueError("champion directory and manifest model disagree")
    if models[selected] != _leaderboard_record(bundle.manifest):
        raise ValueError("leaderboard record does not match champion manifest")
    return bundle
