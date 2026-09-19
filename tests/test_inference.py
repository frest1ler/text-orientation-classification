import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.calibration import BinaryCalibrator
from src.inference import infer_batch, load_champion
from src.models import build_model
from src.registry import ChampionBundle, file_sha256


class MeanLogit(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images.mean(dim=(1, 2, 3))


def test_symmetric_inference_and_calibration() -> None:
    direct = torch.full((2, 3, 2, 2), 2.0)
    rotated = torch.full((2, 3, 2, 2), -1.0)
    calibrator = BinaryCalibrator("temperature", slope=0.5)

    result = infer_batch(
        MeanLogit(), direct, rotated, calibrator, torch.device("cpu")
    )

    expected_direct = 1.0 / (1.0 + np.exp(-2.0))
    expected_rotated = 1.0 / (1.0 + np.exp(1.0))
    expected_symmetric = 0.5 * (expected_direct + 1.0 - expected_rotated)
    assert np.allclose(result["p_direct"], expected_direct)
    assert np.allclose(result["p_rotated"], expected_rotated)
    assert np.allclose(result["p_symmetric"], expected_symmetric)
    assert np.all((0 <= result["p_final"]) & (result["p_final"] <= 1))


def make_small_bundle(tmp_path: Path) -> ChampionBundle:
    model = build_model("small_cnn", dropout=0.0, pretrained=False)
    metrics = {"symmetric": {"brier_score": 0.1}}
    config = {
        "model": {
            "name": "small_cnn",
            "dropout": 0.0,
            "input_height": 32,
            "input_width": 64,
        }
    }
    checkpoint_path = tmp_path / "small.pt"
    torch.save(
        {"model_state": model.state_dict(), "epoch": 2, "metrics": metrics, "config": config},
        checkpoint_path,
    )
    calibration = {
        "checkpoint_epoch": 2,
        "prediction_mode": "symmetric",
        "final_calibrator": {
            "method": "uncalibrated",
            "slope": 1.0,
            "intercept": 0.0,
        },
    }
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    manifest = {
        "model": "small_cnn",
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "calibration": calibration_path.name,
        "calibration_sha256": file_sha256(calibration_path),
        "epoch": 2,
        "metrics": metrics,
        "validation_protocol_sha256": "test",
    }
    return ChampionBundle("small_cnn", tmp_path, checkpoint_path, manifest, calibration)


def test_load_champion_restores_model_and_calibrator(tmp_path: Path) -> None:
    loaded = load_champion(make_small_bundle(tmp_path), torch.device("cpu"))
    output = loaded.model(torch.zeros(2, 3, 32, 64))
    assert output.shape == (2,)
    assert loaded.calibrator.method == "uncalibrated"


def test_load_champion_rejects_manifest_metric_mismatch(tmp_path: Path) -> None:
    bundle = make_small_bundle(tmp_path)
    bundle.manifest["metrics"] = {"symmetric": {"brier_score": 0.2}}
    with pytest.raises(ValueError, match="metrics"):
        load_champion(bundle, torch.device("cpu"))
