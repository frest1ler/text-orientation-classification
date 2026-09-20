"""Protocol-safe comparison of standard and isolated robust champions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.registry import select_champion


def compare_robust(
    registry_dir: str | Path,
    model: str = "mobilenet_v3_large",
    minimum_improvement: float = 0.005,
) -> dict[str, Any]:
    if minimum_improvement < 0:
        raise ValueError("minimum improvement must be non-negative")
    standard = select_champion(registry_dir, model)
    robust = select_champion(registry_dir, f"{model}_robust")
    standard_protocol = standard.manifest["validation_protocol_sha256"]
    robust_protocol = robust.manifest["validation_protocol_sha256"]
    if standard_protocol != robust_protocol:
        raise ValueError("standard and robust champions use incompatible validation protocols")
    standard_brier = float(standard.manifest["metrics"]["symmetric"]["brier_score"])
    robust_brier = float(robust.manifest["metrics"]["symmetric"]["brier_score"])
    improvement = standard_brier - robust_brier
    return {
        "model": model,
        "validation_protocol_sha256": standard_protocol,
        "standard_brier_score": standard_brier,
        "robust_brier_score": robust_brier,
        "brier_improvement": improvement,
        "minimum_required_improvement": minimum_improvement,
        "recommend_robust": improvement >= minimum_improvement,
    }
