import json
from pathlib import Path

import pytest

from src.registry import file_sha256
from src.robust_selection import compare_robust


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_bundle(registry: Path, model: str, brier: float, protocol: str) -> dict:
    directory = registry / "champions" / model
    directory.mkdir(parents=True)
    checkpoint = directory / f"{model}.pt"
    checkpoint.write_bytes(model.encode())
    calibration = directory / "calibration.json"
    write_json(
        calibration,
        {
            "checkpoint_epoch": 1,
            "prediction_mode": "symmetric",
            "final_calibrator": {
                "method": "uncalibrated",
                "slope": 1.0,
                "intercept": 0.0,
            },
        },
    )
    metrics = {
        "symmetric": {
            "brier_score": brier,
            "log_loss": brier + 0.2,
            "roc_auc": 0.9,
            "accuracy": 0.8,
        }
    }
    manifest = {
        "model": model,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": file_sha256(checkpoint),
        "calibration": calibration.name,
        "calibration_sha256": file_sha256(calibration),
        "epoch": 1,
        "metrics": metrics,
        "validation_protocol_sha256": protocol,
    }
    write_json(directory / "champion.json", manifest)
    return {
        key: manifest[key]
        for key in (
            "checkpoint",
            "checkpoint_sha256",
            "calibration",
            "calibration_sha256",
            "epoch",
            "metrics",
            "validation_protocol_sha256",
        )
    }


def prepare(tmp_path: Path, standard_brier: float, robust_brier: float, robust_protocol="p") -> Path:
    registry = tmp_path / "registry"
    standard = make_bundle(registry, "mobilenet_v3_large", standard_brier, "p")
    robust = make_bundle(
        registry / "robust", "mobilenet_v3_large", robust_brier, robust_protocol
    )
    write_json(registry / "leaderboard.json", {"models": {"mobilenet_v3_large": standard}})
    write_json(
        registry / "robust/leaderboard.json",
        {"models": {"mobilenet_v3_large": robust}},
    )
    return registry


def test_robust_requires_material_brier_improvement(tmp_path: Path) -> None:
    registry = prepare(tmp_path, 0.08, 0.074)
    report = compare_robust(registry)
    assert report["brier_improvement"] == pytest.approx(0.006)
    assert report["recommend_robust"] is True


def test_small_improvement_keeps_standard_recommendation(tmp_path: Path) -> None:
    registry = prepare(tmp_path, 0.08, 0.078)
    assert compare_robust(registry)["recommend_robust"] is False


def test_comparison_rejects_different_validation_protocols(tmp_path: Path) -> None:
    registry = prepare(tmp_path, 0.08, 0.07, robust_protocol="different")
    with pytest.raises(ValueError, match="incompatible"):
        compare_robust(registry)
