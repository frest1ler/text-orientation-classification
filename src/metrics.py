"""Binary probability metrics used for validation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


def binary_metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Compute competition and diagnostic metrics from class-1 probabilities."""
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if targets.shape != probabilities.shape:
        raise ValueError("targets and probabilities must have the same shape")
    if targets.size == 0:
        raise ValueError("metrics require at least one sample")
    if not np.isin(targets, (0, 1)).all():
        raise ValueError("targets must contain only 0 and 1")
    if not np.isfinite(probabilities).all() or not ((0 <= probabilities) & (probabilities <= 1)).all():
        raise ValueError("probabilities must be finite values in [0, 1]")

    brier = float(np.mean(np.square(probabilities - targets)))
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    metrics = {
        "brier_score": brier,
        "one_minus_brier": 1.0 - brier,
        "accuracy": float(accuracy_score(targets, probabilities >= 0.5)),
        "log_loss": float(log_loss(targets, clipped, labels=[0, 1])),
    }
    metrics["roc_auc"] = (
        float(roc_auc_score(targets, probabilities)) if np.unique(targets).size == 2 else float("nan")
    )
    return metrics
