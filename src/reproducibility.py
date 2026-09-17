"""Device selection and reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def select_device() -> torch.device:
    """Select CUDA when available and otherwise use CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch and configure deterministic execution."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker from PyTorch's per-worker initial seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Create the seeded generator passed to a PyTorch DataLoader."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
