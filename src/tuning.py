"""Deterministic Optuna search helpers kept separate from the model registry."""

from __future__ import annotations

import csv
import json
import os
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import yaml

from src.config import Config, validate_config


@dataclass(frozen=True)
class TuningBudget:
    train_samples: int
    validation_samples: int
    frozen_epochs: int = 1
    finetune_epochs: int = 3

    def validate(self) -> None:
        if self.train_samples <= 0 or self.validation_samples <= 0:
            raise ValueError("Optuna sample counts must be positive")
        if self.frozen_epochs < 0 or self.finetune_epochs <= 0:
            raise ValueError("Optuna epochs must include positive fine-tuning")


def tuning_root(project_dir: str | Path, model_name: str) -> Path:
    """Return the isolated persistent root for one architecture's search."""
    return Path(project_dir).expanduser() / "tuning" / "optuna" / model_name


def apply_trial_parameters(config: Config, parameters: dict[str, float]) -> Config:
    """Build a validated immutable config from the four allowed parameters."""
    required = {
        "finetune_learning_rate",
        "weight_decay",
        "dropout",
        "symmetry_loss_weight",
    }
    if set(parameters) != required:
        raise ValueError(f"Optuna parameters must be exactly {sorted(required)}")
    tuned = replace(
        config,
        model=replace(config.model, dropout=float(parameters["dropout"])),
        training=replace(
            config.training,
            finetune_learning_rate=float(parameters["finetune_learning_rate"]),
            weight_decay=float(parameters["weight_decay"]),
            symmetry_loss_weight=float(parameters["symmetry_loss_weight"]),
        ),
    )
    validate_config(tuned)
    return tuned


def suggest_parameters(trial: Any) -> dict[str, float]:
    """Use the deliberately small, deadline-friendly search space."""
    return {
        "finetune_learning_rate": trial.suggest_float(
            "finetune_learning_rate", 2e-5, 3e-4, log=True
        ),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 3e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.4),
        "symmetry_loss_weight": trial.suggest_float(
            "symmetry_loss_weight", 0.0, 0.3
        ),
    }


def write_json_atomic(path: str | Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return path


def write_yaml_atomic(path: str | Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    os.replace(temporary, path)
    return path


def lock_search_protocol(path: str | Path, protocol: dict[str, Any]) -> None:
    """Create an immutable study contract or reject incompatible resumption."""
    path = Path(path)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise ValueError(
                "Optuna study protocol differs from the existing study; "
                "use a new study name or directory"
            )
        return
    write_json_atomic(path, protocol)


def promote_trial_checkpoint(
    checkpoint: str | Path,
    destination_dir: str | Path,
    trial_number: int,
    value: float,
    parameters: dict[str, float],
) -> Path:
    """Atomically replace the tuning-only best checkpoint and its metadata."""
    checkpoint, destination = Path(checkpoint), Path(destination_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "best.pt"
    temporary = destination / ".best.pt.tmp"
    shutil.copy2(checkpoint, temporary)
    os.replace(temporary, target)
    write_json_atomic(
        destination / "best_trial.json",
        {
            "trial": trial_number,
            "symmetric_brier_score": value,
            "parameters": parameters,
        },
    )
    return target


def export_trials_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    """Write a stable compact trial table without a pandas dependency."""
    path = Path(path)
    rows = list(rows)
    columns = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
    return path


def confirmation_config(config: Config, best_parameters: dict[str, float]) -> dict[str, Any]:
    """Return the full-data config used to confirm, calibrate, then promote."""
    return asdict(apply_trial_parameters(config, best_parameters))
