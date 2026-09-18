"""Leakage-resistant probability calibration for binary predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.metrics import binary_metrics


EPSILON = 1e-7


def probabilities_to_logits(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), EPSILON, 1 - EPSILON)
    return np.log(probabilities / (1.0 - probabilities))


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


@dataclass(frozen=True)
class BinaryCalibrator:
    method: str
    slope: float = 1.0
    intercept: float = 0.0

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        logits = probabilities_to_logits(probabilities)
        return sigmoid(self.slope * logits + self.intercept)

    def to_dict(self) -> dict[str, float | str]:
        return {
            "method": self.method,
            "slope": self.slope,
            "intercept": self.intercept,
        }


def _fit_temperature(probabilities: np.ndarray, targets: np.ndarray) -> BinaryCalibrator:
    logits = probabilities_to_logits(probabilities)

    def loss(log_temperature: float) -> float:
        calibrated = sigmoid(logits / np.exp(log_temperature))
        clipped = np.clip(calibrated, EPSILON, 1 - EPSILON)
        return float(-np.mean(targets * np.log(clipped) + (1 - targets) * np.log(1 - clipped)))

    left, right = -4.0, 4.0
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    x1, x2 = right - ratio * (right - left), left + ratio * (right - left)
    f1, f2 = loss(x1), loss(x2)
    for _ in range(80):
        if f1 < f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = loss(x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = loss(x2)
    temperature = float(np.exp((left + right) / 2.0))
    return BinaryCalibrator("temperature", slope=1.0 / temperature)


def _fit_platt(probabilities: np.ndarray, targets: np.ndarray) -> BinaryCalibrator:
    logits = probabilities_to_logits(probabilities).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", random_state=0)
    model.fit(logits, targets)
    return BinaryCalibrator(
        "platt", slope=float(model.coef_[0, 0]), intercept=float(model.intercept_[0])
    )


FITTERS: dict[str, Callable[[np.ndarray, np.ndarray], BinaryCalibrator]] = {
    "temperature": _fit_temperature,
    "platt": _fit_platt,
}


def evaluate_oof_calibration(
    targets: np.ndarray,
    probabilities: np.ndarray,
    folds: int = 5,
    seed: int = 42,
) -> tuple[BinaryCalibrator, dict[str, Any], dict[str, np.ndarray]]:
    """Select a calibrator using predictions from folds it did not train on."""
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    binary_metrics(targets, probabilities)
    class_counts = np.bincount(targets, minlength=2)
    if folds < 2 or class_counts.min() < folds:
        raise ValueError("each class must contain at least one sample per calibration fold")

    predictions = {"uncalibrated": probabilities.copy()}
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for method, fitter in FITTERS.items():
        oof = np.empty_like(probabilities)
        for train_indices, holdout_indices in splitter.split(probabilities, targets):
            calibrator = fitter(probabilities[train_indices], targets[train_indices])
            oof[holdout_indices] = calibrator.predict(probabilities[holdout_indices])
        predictions[method] = oof

    metrics = {name: binary_metrics(targets, values) for name, values in predictions.items()}
    selected = min(metrics, key=lambda name: metrics[name]["brier_score"])
    final = (
        BinaryCalibrator("uncalibrated")
        if selected == "uncalibrated"
        else FITTERS[selected](probabilities, targets)
    )
    report = {
        "protocol": "stratified_out_of_fold",
        "folds": folds,
        "seed": seed,
        "selected_method": selected,
        "oof_metrics": metrics,
        "final_calibrator": final.to_dict(),
    }
    return final, report, predictions
