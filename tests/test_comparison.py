import csv
import json
from pathlib import Path

import pytest

from src.comparison import compare_champions, write_comparison
from src.registry import file_sha256


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_bundle(registry: Path, model: str, brier: float, protocol: str) -> dict:
    directory = registry / "champions" / model
    directory.mkdir(parents=True)
    checkpoint = directory / f"{model}_0.800000.pt"
    checkpoint.write_bytes(f"checkpoint:{model}".encode())
    calibration = directory / "calibration.json"
    _write_json(
        calibration,
        {
            "checkpoint_epoch": 4,
            "prediction_mode": "symmetric",
            "final_calibrator": {
                "method": "temperature",
                "slope": 0.5,
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
        "epoch": 4,
        "metrics": metrics,
        "validation_protocol_sha256": protocol,
    }
    _write_json(directory / "champion.json", manifest)
    return {key: value for key, value in manifest.items() if key != "model"}


def _make_registry(tmp_path, second_protocol="protocol"):
    registry = tmp_path / "registry"
    mobile = _make_bundle(registry, "mobilenet_v3_large", 0.08, "protocol")
    efficient = _make_bundle(
        registry, "efficientnet_b0", 0.07, second_protocol
    )
    _write_json(
        registry / "leaderboard.json",
        {"models": {"mobilenet_v3_large": mobile, "efficientnet_b0": efficient}},
    )
    return registry


def test_comparison_ranks_by_locked_champion_metric_and_writes_reports(tmp_path) -> None:
    registry = _make_registry(tmp_path)

    report = compare_champions(
        registry, required_models=("mobilenet_v3_large", "efficientnet_b0")
    )
    paths = write_comparison(tmp_path / "reports", report)

    assert report["winner"] == "efficientnet_b0"
    assert [record["rank"] for record in report["ranking"]] == [1, 2]
    assert all(path.is_file() for path in paths)
    with paths[1].open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["model"] for row in rows] == [
        "efficientnet_b0",
        "mobilenet_v3_large",
    ]
    assert json.loads(paths[0].read_text())["status"] == "comparable"


def test_comparison_requires_requested_models(tmp_path) -> None:
    registry = _make_registry(tmp_path)

    with pytest.raises(ValueError, match="missing.*vit_b_16"):
        compare_champions(registry, required_models=("vit_b_16",))


def test_comparison_rejects_mixed_validation_protocols(tmp_path) -> None:
    registry = _make_registry(tmp_path, second_protocol="other")

    with pytest.raises(ValueError, match="incompatible"):
        compare_champions(registry)
