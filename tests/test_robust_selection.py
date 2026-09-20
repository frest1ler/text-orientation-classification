from pathlib import Path

import pytest

from src.robust_selection import compare_robust
from tests.test_registry import make_bundle, write_json


def prepare(tmp_path: Path, standard_brier: float, robust_brier: float, robust_protocol="p") -> Path:
    registry = tmp_path / "registry"
    standard = make_bundle(registry, "mobilenet_v3_large", standard_brier, "p")
    robust = make_bundle(
        registry / "robust", "mobilenet_v3_large", robust_brier, robust_protocol
    )
    write_json(registry / "leaderboard.json", {"models": {"mobilenet_v3_large": standard}})
    write_json(
        registry / "robust/leaderboard.json",
        {"models": {"mobilenet_v3_large": robust}},
    )
    return registry


def test_robust_requires_material_brier_improvement(tmp_path: Path) -> None:
    registry = prepare(tmp_path, 0.08, 0.074)
    report = compare_robust(registry)
    assert report["brier_improvement"] == pytest.approx(0.006)
    assert report["recommend_robust"] is True


def test_small_improvement_keeps_standard_recommendation(tmp_path: Path) -> None:
    registry = prepare(tmp_path, 0.08, 0.078)
    assert compare_robust(registry)["recommend_robust"] is False


def test_comparison_rejects_different_validation_protocols(tmp_path: Path) -> None:
    registry = prepare(tmp_path, 0.08, 0.07, robust_protocol="different")
    with pytest.raises(ValueError, match="incompatible"):
        compare_robust(registry)
