import json
from pathlib import Path

import pytest

from src.project_layout import ProjectLayout, resolve_artifact_root
from src.registry import file_sha256, select_champion


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_bundle(registry: Path, model: str, brier: float) -> dict:
    directory = registry / "champions" / model
    directory.mkdir(parents=True)
    checkpoint = directory / f"{model}_0.800000.pt"
    checkpoint.write_bytes(f"checkpoint:{model}".encode())
    calibration = directory / "calibration.json"
    write_json(
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
        "validation_protocol_sha256": "protocol",
    }
    write_json(directory / "champion.json", manifest)
    return {
        "checkpoint": manifest["checkpoint"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "calibration": manifest["calibration"],
        "calibration_sha256": manifest["calibration_sha256"],
        "metrics": metrics,
        "epoch": 4,
        "validation_protocol_sha256": "protocol",
    }


def test_registry_selects_best_or_named_model(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    mobile = make_bundle(registry, "mobilenet_v3_large", 0.08)
    efficient = make_bundle(registry, "efficientnet_b0", 0.07)
    write_json(
        registry / "leaderboard.json",
        {"models": {"mobilenet_v3_large": mobile, "efficientnet_b0": efficient}},
    )

    assert select_champion(registry, "best").model == "efficientnet_b0"
    assert select_champion(registry, "mobilenet_v3_large").model == "mobilenet_v3_large"


def test_uploaded_single_bundle_supports_best_selection(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    make_bundle(registry, "mobilenet_v3_large", 0.08)
    bundle_dir = registry / "champions/mobilenet_v3_large"

    assert select_champion(bundle_dir, "best").model == "mobilenet_v3_large"
    with pytest.raises(ValueError, match="not requested"):
        select_champion(bundle_dir, "efficientnet_b0")


def test_registry_rejects_modified_checkpoint(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    record = make_bundle(registry, "mobilenet_v3_large", 0.08)
    write_json(registry / "leaderboard.json", {"models": {"mobilenet_v3_large": record}})
    checkpoint = registry / "champions/mobilenet_v3_large" / record["checkpoint"]
    checkpoint.write_bytes(b"modified")

    with pytest.raises(ValueError, match="SHA-256"):
        select_champion(registry)


def test_project_layout_and_artifact_sources(tmp_path: Path) -> None:
    layout = ProjectLayout.from_root(tmp_path / "project")
    layout.create_output_directories()
    assert layout.registry.is_dir()
    assert layout.training_runs.is_dir()
    assert resolve_artifact_root(layout.root, "drive") == layout.registry
    upload = tmp_path / "upload"
    assert resolve_artifact_root(layout.root, "upload", upload) == upload
    with pytest.raises(ValueError, match="uploaded_path"):
        resolve_artifact_root(layout.root, "upload")
