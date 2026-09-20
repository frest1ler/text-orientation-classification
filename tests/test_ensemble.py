import json
from pathlib import Path

import numpy as np
import pytest

from src.ensemble import (
    EnsembleCandidate,
    build_ensemble_manifest,
    combine_prediction_rows,
    promote_ensemble,
    search_ensemble,
    validate_candidates,
)
from src.registry import ChampionBundle


def _candidate(
    selector: str,
    probabilities: list[float],
    *,
    protocol: str = "protocol",
    sample_ids: list[str] | None = None,
    targets: list[int] | None = None,
) -> EnsembleCandidate:
    targets = targets or [0, 0, 1, 1]
    sample_ids = sample_ids or ["0", "1", "2", "3"]
    manifest = {
        "model": selector.removesuffix("_robust"),
        "checkpoint_sha256": f"checkpoint-{selector}",
        "calibration_sha256": f"calibration-{selector}",
        "validation_protocol_sha256": protocol,
    }
    bundle = ChampionBundle(selector, Path("."), Path("model.pt"), manifest, {})
    return EnsembleCandidate(
        selector,
        bundle,
        np.asarray(targets),
        np.asarray(sample_ids),
        np.asarray(probabilities),
        "oof_temperature",
    )


def test_search_ensemble_finds_deterministic_midpoint() -> None:
    candidates = [
        _candidate("first", [0.0, 0.0, 0.6, 0.6]),
        _candidate("second", [0.4, 0.4, 1.0, 1.0]),
    ]

    result = search_ensemble(candidates, 0.25)

    assert result["weights"] == [0.5, 0.5]
    assert result["evaluated"] == 3
    assert result["metrics"]["brier_score"] == pytest.approx(0.04)


@pytest.mark.parametrize(
    "second, message",
    [
        (_candidate("second", [0.1] * 4, protocol="other"), "protocol"),
        (
            _candidate("second", [0.1] * 4, sample_ids=["1", "0", "2", "3"]),
            "order",
        ),
        (_candidate("second", [0.1] * 4, targets=[0, 1, 0, 1]), "targets"),
    ],
)
def test_candidate_validation_rejects_incompatible_inputs(
    second: EnsembleCandidate, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_candidates([_candidate("first", [0.2] * 4), second])


def test_manifest_promotion_is_isolated_and_keeps_better(tmp_path: Path) -> None:
    candidates = [
        _candidate("first", [0.0, 0.0, 0.6, 0.6]),
        _candidate("second", [0.4, 0.4, 1.0, 1.0]),
    ]
    search = search_ensemble(candidates, 0.25)
    manifest = build_ensemble_manifest(candidates, search)

    promoted = promote_ensemble(manifest, tmp_path / "registry")
    worse = json.loads(json.dumps(manifest))
    worse["validation_metrics"]["brier_score"] += 0.1
    kept = promote_ensemble(worse, tmp_path / "registry")

    assert promoted["status"] == "promoted"
    assert kept["status"] == "kept_existing"
    assert not (tmp_path / "registry/leaderboard.json").exists()
    assert (tmp_path / "registry/ensembles/leaderboard.json").is_file()


def test_ensemble_that_does_not_improve_a_component_is_not_promoted(
    tmp_path: Path,
) -> None:
    candidates = [
        _candidate("strong", [0.0, 0.0, 1.0, 1.0]),
        _candidate("weak", [0.4, 0.4, 0.6, 0.6]),
    ]
    manifest = build_ensemble_manifest(candidates, search_ensemble(candidates, 0.25))

    result = promote_ensemble(manifest, tmp_path / "registry")

    assert result["status"] == "not_better_than_component"
    assert not (tmp_path / "registry/ensembles").exists()


def test_combine_prediction_rows_checks_order_and_weights() -> None:
    first = [
        {"image_id": "a", "p_direct": 0.2, "p_rotated": 0.7, "p_symmetric": 0.25, "p_final": 0.3},
        {"image_id": "b", "p_direct": 0.8, "p_rotated": 0.1, "p_symmetric": 0.85, "p_final": 0.9},
    ]
    second = [
        {"image_id": "a", "p_direct": 0.4, "p_rotated": 0.5, "p_symmetric": 0.45, "p_final": 0.5},
        {"image_id": "b", "p_direct": 0.6, "p_rotated": 0.3, "p_symmetric": 0.65, "p_final": 0.7},
    ]

    result = combine_prediction_rows([first, second], [0.25, 0.75])

    assert result[0]["p_final"] == pytest.approx(0.45)
    assert result[1]["p_final"] == pytest.approx(0.75)
    reversed_second = list(reversed(second))
    with pytest.raises(ValueError, match="order"):
        combine_prediction_rows([first, reversed_second], [0.5, 0.5])
    with pytest.raises(ValueError, match="sum"):
        combine_prediction_rows([first, second], [0.4, 0.4])
