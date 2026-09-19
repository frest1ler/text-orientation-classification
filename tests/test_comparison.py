import csv
import json

import pytest

from src.comparison import compare_champions, write_comparison
from tests.test_registry import make_bundle, write_json


def _make_registry(tmp_path, second_protocol="protocol"):
    registry = tmp_path / "registry"
    mobile = make_bundle(registry, "mobilenet_v3_large", 0.08)
    efficient = make_bundle(
        registry, "efficientnet_b0", 0.07, protocol=second_protocol
    )
    write_json(
        registry / "leaderboard.json",
        {"models": {"mobilenet_v3_large": mobile, "efficientnet_b0": efficient}},
    )
    return registry


def test_comparison_ranks_by_locked_champion_metric_and_writes_reports(tmp_path) -> None:
    registry = _make_registry(tmp_path)

    report = compare_champions(
        registry, required_models=("mobilenet_v3_large", "efficientnet_b0")
    )
    paths = write_comparison(tmp_path / "reports", report)

    assert report["winner"] == "efficientnet_b0"
    assert [record["rank"] for record in report["ranking"]] == [1, 2]
    assert all(path.is_file() for path in paths)
    with paths[1].open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["model"] for row in rows] == [
        "efficientnet_b0",
        "mobilenet_v3_large",
    ]
    assert json.loads(paths[0].read_text())["status"] == "comparable"


def test_comparison_requires_requested_models(tmp_path) -> None:
    registry = _make_registry(tmp_path)

    with pytest.raises(ValueError, match="missing.*vit_b_16"):
        compare_champions(registry, required_models=("vit_b_16",))


def test_comparison_rejects_mixed_validation_protocols(tmp_path) -> None:
    registry = _make_registry(tmp_path, second_protocol="other")

    with pytest.raises(ValueError, match="incompatible"):
        compare_champions(registry)
