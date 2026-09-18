"""Minimal reusable binary training loop and checkpoint handling."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.metrics import binary_metrics


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool = True,
) -> float:
    """Train for one epoch and return mean BCE loss per example."""
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    loss_sum = 0.0
    sample_count = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, dtype=torch.float32, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        context = torch.autocast(device_type=device.type, enabled=True) if use_amp else nullcontext()
        with context:
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = targets.shape[0]
        loss_sum += float(loss.detach()) * batch_size
        sample_count += batch_size
    if sample_count == 0:
        raise ValueError("training loader is empty")
    return loss_sum / sample_count


@torch.inference_mode()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    """Evaluate calibrated class-1 probabilities and return binary metrics."""
    model.eval()
    all_targets: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    for images, targets in loader:
        logits = model(images.to(device, non_blocking=True))
        all_probabilities.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    if not all_targets:
        raise ValueError("validation loader is empty")
    return binary_metrics(np.concatenate(all_targets), np.concatenate(all_probabilities))


def paired_probabilities(
    logits_direct: torch.Tensor, logits_rotated: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return direct, rotated, and symmetry-enforced class-1 probabilities."""
    probabilities_direct = torch.sigmoid(logits_direct)
    probabilities_rotated = torch.sigmoid(logits_rotated)
    probabilities_symmetric = 0.5 * (
        probabilities_direct + 1.0 - probabilities_rotated
    )
    return probabilities_direct, probabilities_rotated, probabilities_symmetric


def train_paired_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    symmetry_loss_weight: float,
    amp: bool = True,
) -> dict[str, float]:
    """Train on explicit orientation pairs using shared model weights."""
    if symmetry_loss_weight < 0:
        raise ValueError("symmetry_loss_weight must be non-negative")
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    totals = {"loss": 0.0, "classification_loss": 0.0, "symmetry_loss": 0.0}
    sample_count = 0
    for batch in loader:
        direct = batch["image"].to(device, non_blocking=True)
        rotated = batch["rotated"].to(device, non_blocking=True)
        targets = batch["target"].to(device, dtype=torch.float32, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        context = torch.autocast(device_type=device.type, enabled=True) if use_amp else nullcontext()
        with context:
            logits = model(torch.cat((direct, rotated), dim=0))
            logits_direct, logits_rotated = logits.chunk(2)
            direct_loss = criterion(logits_direct, targets)
            rotated_loss = criterion(logits_rotated, 1.0 - targets)
            classification_loss = 0.5 * (direct_loss + rotated_loss)
            probabilities_direct, probabilities_rotated, _ = paired_probabilities(
                logits_direct, logits_rotated
            )
            symmetry_loss = torch.square(
                probabilities_direct + probabilities_rotated - 1.0
            ).mean()
            loss = classification_loss + symmetry_loss_weight * symmetry_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = targets.shape[0]
        totals["loss"] += float(loss.detach()) * batch_size
        totals["classification_loss"] += float(classification_loss.detach()) * batch_size
        totals["symmetry_loss"] += float(symmetry_loss.detach()) * batch_size
        sample_count += batch_size
    if sample_count == 0:
        raise ValueError("training loader is empty")
    return {name: value / sample_count for name, value in totals.items()}


@torch.inference_mode()
def predict_paired(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, np.ndarray]:
    """Collect targets and direct/symmetric probabilities on paired data."""
    model.eval()
    targets_all: list[np.ndarray] = []
    direct_all: list[np.ndarray] = []
    symmetric_all: list[np.ndarray] = []
    symmetry_errors: list[np.ndarray] = []
    for batch in loader:
        direct = batch["image"].to(device, non_blocking=True)
        rotated = batch["rotated"].to(device, non_blocking=True)
        logits = model(torch.cat((direct, rotated), dim=0))
        logits_direct, logits_rotated = logits.chunk(2)
        probabilities_direct, probabilities_rotated, probabilities_symmetric = (
            paired_probabilities(logits_direct, logits_rotated)
        )
        targets_all.append(batch["target"].numpy())
        direct_all.append(probabilities_direct.cpu().numpy())
        symmetric_all.append(probabilities_symmetric.cpu().numpy())
        symmetry_errors.append(
            torch.abs(probabilities_direct + probabilities_rotated - 1.0).cpu().numpy()
        )
    if not targets_all:
        raise ValueError("validation loader is empty")
    return {
        "targets": np.concatenate(targets_all),
        "direct": np.concatenate(direct_all),
        "symmetric": np.concatenate(symmetric_all),
        "symmetry_error": np.concatenate(symmetry_errors),
    }


@torch.inference_mode()
def evaluate_paired(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, Any]:
    """Compare direct and symmetry-enforced predictions on paired data."""
    predictions = predict_paired(model, loader, device)
    return {
        "direct": binary_metrics(predictions["targets"], predictions["direct"]),
        "symmetric": binary_metrics(predictions["targets"], predictions["symmetric"]),
        "mean_symmetry_error": float(predictions["symmetry_error"].mean()),
    }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    config: dict[str, Any],
) -> None:
    """Atomically save model state and reproducibility metadata."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": config,
        },
        temporary_path,
    )
    temporary_path.replace(checkpoint_path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a trusted local checkpoint and restore model/optimizer state."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return {
        "epoch": checkpoint["epoch"],
        "metrics": checkpoint["metrics"],
        "config": checkpoint["config"],
    }
