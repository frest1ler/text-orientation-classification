import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.calibration import BinaryCalibrator
from src.inference import infer_batch, load_champion, resolve_inference_batch_size
from src.models import build_model
from src.registry import ChampionBundle, file_sha256


class MeanLogit(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images.mean(dim=(1, 2, 3))


def test_inference_batch_defaults_to_selected_champion_config() -> None:
    config = {"inference": {"batch_size": 32}}

    assert resolve_inference_batch_size(config) == 32
    assert resolve_inference_batch_size(config, override=8) == 8
    with pytest.raises(ValueError, match="positive"):
        resolve_inference_batch_size(config, override=0)


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


def test_load_vit_champion_reconstructs_rectangular_dimensions(
    tmp_path: Path, monkeypatch
) -> None:
    dimensions = {}

    class TinyContract(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(self, images):
            return images.mean(dim=(1, 2, 3)) * self.weight

    def fake_build_model(name, **kwargs):
        dimensions.update({"name": name, **kwargs})
        return TinyContract()

    monkeypatch.setattr("src.inference.build_model", fake_build_model)
    metrics = {"symmetric": {"brier_score": 0.08}}
    config = {
        "model": {
            "name": "vit_b_16",
            "dropout": 0.1,
            "input_height": 96,
            "input_width": 384,
        }
    }
    checkpoint = tmp_path / "vit.pt"
    torch.save(
        {
            "model_state": TinyContract().state_dict(),
            "epoch": 3,
            "metrics": metrics,
            "config": config,
        },
        checkpoint,
    )
    calibration = {
        "checkpoint_epoch": 3,
        "prediction_mode": "symmetric",
        "final_calibrator": {
            "method": "temperature",
            "slope": 0.7,
            "intercept": 0.0,
        },
    }
    bundle = ChampionBundle(
        "vit_b_16",
        tmp_path,
        checkpoint,
        {"model": "vit_b_16", "epoch": 3, "metrics": metrics},
        calibration,
    )

    loaded = load_champion(bundle, torch.device("cpu"))

    assert dimensions == {
        "name": "vit_b_16",
        "dropout": 0.1,
        "pretrained": False,
        "input_height": 96,
        "input_width": 384,
    }
    assert loaded.calibrator.slope == pytest.approx(0.7)
    assert loaded.model(torch.ones(1, 3, 2, 2)).shape == (1,)
