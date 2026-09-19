"""Validated champion loading and symmetric test inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from src.calibration import BinaryCalibrator
from src.models import build_model
from src.registry import ChampionBundle
from src.test_data import ZipTestDataset
from src.training import paired_probabilities
from src.transforms import build_preprocess


@dataclass(frozen=True)
class LoadedChampion:
    bundle: ChampionBundle
    model: nn.Module
    config: dict[str, Any]
    calibrator: BinaryCalibrator


def load_champion(bundle: ChampionBundle, device: torch.device) -> LoadedChampion:
    """Load a hash-validated champion and verify its embedded metadata."""
    checkpoint = torch.load(bundle.checkpoint_path, map_location="cpu", weights_only=True)
    required = {"model_state", "epoch", "metrics", "config"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"champion checkpoint is missing required fields: {missing}")
    if checkpoint["epoch"] != bundle.manifest["epoch"]:
        raise ValueError("checkpoint epoch does not match champion manifest")
    if checkpoint["metrics"] != bundle.manifest["metrics"]:
        raise ValueError("checkpoint metrics do not match champion manifest")
    config = checkpoint["config"]
    model_config = config.get("model", {})
    if model_config.get("name") != bundle.model:
        raise ValueError("checkpoint model does not match champion manifest")
    final = bundle.calibration["final_calibrator"]
    calibrator = BinaryCalibrator(
        method=final["method"],
        slope=float(final["slope"]),
        intercept=float(final["intercept"]),
    )
    model = build_model(
        bundle.model,
        dropout=float(model_config["dropout"]),
        pretrained=False,
        input_height=int(model_config["input_height"]),
        input_width=int(model_config["input_width"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return LoadedChampion(bundle, model, config, calibrator)


def make_test_loader(
    zip_path: str,
    config: dict[str, Any],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    start_index: int = 0,
    end_index: int | None = None,
) -> tuple[ZipTestDataset, DataLoader]:
    """Build an ordered, augmentation-free test loader from the training config."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    model_config = config["model"]
    data_config = config["data"]
    preprocess = build_preprocess(
        model_config["name"],
        int(model_config["input_height"]),
        int(model_config["input_width"]),
    )
    dataset = ZipTestDataset(
        zip_path,
        data_config["image_prefix"],
        data_config["sample_submission_member"],
        preprocess,
    )
    end_index = len(dataset) if end_index is None else end_index
    if not 0 <= start_index <= end_index <= len(dataset):
        raise ValueError("start_index is outside the test dataset")
    loader_dataset = Subset(dataset, range(start_index, end_index))
    loader = DataLoader(
        loader_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return dataset, loader


@torch.inference_mode()
def infer_batch(
    model: nn.Module,
    direct: torch.Tensor,
    rotated: torch.Tensor,
    calibrator: BinaryCalibrator,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run direct/rotated inference and return calibrated symmetric probabilities."""
    logits = model(
        torch.cat(
            (
                direct.to(device, non_blocking=True),
                rotated.to(device, non_blocking=True),
            ),
            dim=0,
        )
    )
    logits_direct, logits_rotated = logits.chunk(2)
    probabilities_direct, probabilities_rotated, probabilities_symmetric = (
        paired_probabilities(logits_direct, logits_rotated)
    )
    direct_values = probabilities_direct.cpu().numpy().astype(np.float64)
    rotated_values = probabilities_rotated.cpu().numpy().astype(np.float64)
    symmetric_values = probabilities_symmetric.cpu().numpy().astype(np.float64)
    final_values = calibrator.predict(symmetric_values)
    arrays = {
        "p_direct": direct_values,
        "p_rotated": rotated_values,
        "p_symmetric": symmetric_values,
        "symmetry_error": np.abs(direct_values + rotated_values - 1.0),
        "p_final": final_values,
    }
    if not all(np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("inference produced non-finite values")
    for name in ("p_direct", "p_rotated", "p_symmetric", "p_final"):
        if not ((0.0 <= arrays[name]) & (arrays[name] <= 1.0)).all():
            raise ValueError(f"{name} contains probabilities outside [0, 1]")
    return arrays
