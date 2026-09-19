import json
from pathlib import Path

import torch

from src.champions import promote_champion
from src.config import load_config


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def make_run(
    root: Path,
    name: str,
    brier: float,
    accuracy: float,
    validation_samples: int = 512,
) -> Path:
    run_dir = root / name
    run_dir.mkdir()
    config = load_config("configs/baseline.yaml").to_dict()
    runtime = {
        "pretrained": True,
        "train_base_samples": 2048,
        "validation_base_samples": validation_samples,
        "num_workers": 0,
        "batch_size": 8,
        "validation_batch_size": 8,
        "frozen_epochs": 1,
        "finetune_epochs": 1,
        "run_name": name,
        "device": "cpu",
    }
    metrics = {
        "direct": {
            "brier_score": brier + 0.01,
            "one_minus_brier": 0.99 - brier,
            "accuracy": accuracy - 0.01,
            "log_loss": 0.6,
            "roc_auc": 0.7,
        },
        "symmetric": {
            "brier_score": brier,
            "one_minus_brier": 1.0 - brier,
            "accuracy": accuracy,
            "log_loss": brier + 0.3,
            "roc_auc": accuracy + 0.05,
        },
        "mean_symmetry_error": 0.1,
    }
    history_record = {
        "epoch": 2,
        "phase": "finetune",
        "loss": 0.5,
        "classification_loss": 0.49,
        "symmetry_loss": 0.1,
        **metrics,
    }
    torch.save(
        {
            "model_state": {"weight": torch.tensor([brier])},
            "optimizer_state": {},
            "epoch": 2,
            "metrics": metrics,
            "config": config,
        },
        run_dir / "best.pt",
    )
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "runtime.json", runtime)
    write_json(run_dir / "history.json", [history_record])
    write_json(run_dir / "environment.json", {"git_commit": "test", "gpu": "cpu"})
    write_json(
        run_dir / "calibration.json",
        {
            "checkpoint_epoch": 2,
            "prediction_mode": "symmetric",
            "selected_method": "temperature",
            "final_calibrator": {
                "method": "temperature",
                "slope": 0.8,
                "intercept": 0.0,
            },
        },
    )
    return run_dir


def test_first_candidate_becomes_model_champion(tmp_path: Path) -> None:
    run = make_run(tmp_path, "first", brier=0.20, accuracy=0.70)
    registry = tmp_path / "registry"

    result = promote_champion(run, registry)

    assert result["status"] == "promoted"
    champion = json.loads(
        (registry / "champions/mobilenet_v3_large/champion.json").read_text()
    )
    assert champion["checkpoint"] == "mobilenet_v3_large_0.700000.pt"
    assert (registry / "champions/mobilenet_v3_large" / champion["checkpoint"]).is_file()
    assert (registry / "champions/mobilenet_v3_large/calibration.json").is_file()
    assert champion["calibration"] == "calibration.json"
    assert json.loads((registry / "leaderboard.json").read_text())["models"]


def test_worse_candidate_keeps_existing_champion(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    first = make_run(tmp_path, "first", brier=0.20, accuracy=0.70)
    worse = make_run(tmp_path, "worse", brier=0.22, accuracy=0.80)
    promote_champion(first, registry)

    result = promote_champion(worse, registry)

    assert result["status"] == "kept_existing"
    assert result["champion"] == "mobilenet_v3_large_0.700000.pt"


def test_better_candidate_replaces_only_model_champion(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    first = make_run(tmp_path, "first", brier=0.20, accuracy=0.70)
    better = make_run(tmp_path, "better", brier=0.18, accuracy=0.75)
    promote_champion(first, registry)

    result = promote_champion(better, registry)

    model_dir = registry / "champions/mobilenet_v3_large"
    assert result["status"] == "promoted"
    assert (model_dir / "mobilenet_v3_large_0.750000.pt").is_file()
    assert not (model_dir / "mobilenet_v3_large_0.700000.pt").exists()


def test_incompatible_validation_does_not_replace_champion(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    first = make_run(tmp_path, "first", brier=0.20, accuracy=0.70)
    incompatible = make_run(
        tmp_path, "incompatible", brier=0.10, accuracy=0.90, validation_samples=256
    )
    promote_champion(first, registry)

    result = promote_champion(incompatible, registry)

    assert result["status"] == "incompatible_validation_protocol"
    champion = json.loads(
        (registry / "champions/mobilenet_v3_large/champion.json").read_text()
    )
    assert champion["checkpoint"] == "mobilenet_v3_large_0.700000.pt"


def test_same_run_repairs_legacy_bundle_without_calibration(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    run = make_run(tmp_path, "first", brier=0.20, accuracy=0.70)
    promote_champion(run, registry)
    model_dir = registry / "champions/mobilenet_v3_large"
    manifest_path = model_dir / "champion.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("calibration")
    manifest.pop("calibration_sha256")
    write_json(manifest_path, manifest)
    (model_dir / "calibration.json").unlink()

    result = promote_champion(run, registry)

    repaired = json.loads(manifest_path.read_text())
    assert result["status"] == "promoted"
    assert repaired["calibration"] == "calibration.json"
    assert (model_dir / "calibration.json").is_file()
