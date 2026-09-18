"""Epoch-boundary recovery checkpoints for resumable training."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from torch import nn


RECOVERY_PROTOCOL_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECOVERY_SOURCES = (
    "scripts/train.py",
    "src/models.py",
    "src/synthetic.py",
    "src/training.py",
    "src/transforms.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recovery_fingerprint(config: dict[str, Any], runtime: dict[str, Any]) -> str:
    """Hash every setting that changes the resumed optimization trajectory."""
    payload = {
        "protocol_version": RECOVERY_PROTOCOL_VERSION,
        "config": config,
        "source_sha256": {
            relative: _sha256(PROJECT_ROOT / relative) for relative in RECOVERY_SOURCES
        },
        "runtime": {
            key: runtime[key]
            for key in (
                "pretrained",
                "train_base_samples",
                "validation_base_samples",
                "num_workers",
                "batch_size",
                "validation_batch_size",
                "frozen_epochs",
                "finetune_epochs",
            )
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def copy_file_atomic(source: str | Path, destination: str | Path) -> Path:
    """Copy a file without exposing a partially written destination."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
    return destination


def save_recovery(
    directory: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state: dict[str, Any],
    fingerprint: str,
    train_generator: torch.Generator,
    completed: bool = False,
) -> Path:
    """Atomically persist a recovery checkpoint and human-readable status."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "last.pt"
    temporary = directory / ".last.pt.tmp"
    payload = {
        "protocol_version": RECOVERY_PROTOCOL_VERSION,
        "fingerprint": fingerprint,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "state": state,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "train_generator_state": train_generator.get_state(),
        "completed": completed,
    }
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    _atomic_json(
        directory / "status.json",
        {
            "protocol_version": RECOVERY_PROTOCOL_VERSION,
            "fingerprint": fingerprint,
            "phase": state["phase"],
            "epoch": state["epoch"],
            "phase_epoch": state["phase_epoch"],
            "best_brier": state["best_brier"],
            "completed": completed,
        },
    )
    return destination


def inspect_recovery(
    directory: str | Path, expected_fingerprint: str
) -> dict[str, Any]:
    """Read authoritative resume metadata from the atomic checkpoint."""
    checkpoint = torch.load(
        Path(directory) / "last.pt", map_location="cpu", weights_only=True
    )
    if checkpoint.get("protocol_version") != RECOVERY_PROTOCOL_VERSION:
        raise ValueError("recovery protocol version is incompatible")
    if checkpoint.get("fingerprint") != expected_fingerprint:
        raise ValueError("recovery checkpoint is incompatible with this run")
    return {**checkpoint["state"], "completed": bool(checkpoint["completed"])}


def load_recovery(
    directory: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_fingerprint: str,
    train_generator: torch.Generator,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Restore a compatible checkpoint, optimizer, and random streams."""
    path = Path(directory) / "last.pt"
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if checkpoint.get("protocol_version") != RECOVERY_PROTOCOL_VERSION:
        raise ValueError("recovery protocol version is incompatible")
    if checkpoint.get("fingerprint") != expected_fingerprint:
        raise ValueError("recovery checkpoint is incompatible with this run")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and checkpoint["cuda_rng_states"]:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
    train_generator.set_state(checkpoint["train_generator_state"])
    return {
        **checkpoint["state"],
        "completed": bool(checkpoint["completed"]),
    }
