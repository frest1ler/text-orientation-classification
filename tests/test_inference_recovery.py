from pathlib import Path

import pytest

from src.inference_recovery import (
    inference_fingerprint,
    load_inference_recovery,
    save_inference_recovery,
)


def row(identifier: str) -> dict:
    return {
        "image_id": identifier,
        "p_direct": 0.2,
        "p_rotated": 0.8,
        "p_symmetric": 0.2,
        "symmetry_error": 0.0,
        "p_final": 0.25,
    }


def test_recovery_round_trip_and_committed_prefix(tmp_path: Path) -> None:
    fingerprint = inference_fingerprint({"model": "small"})
    identifiers = ["a", "b", "c"]
    save_inference_recovery(
        tmp_path, fingerprint, [row("a"), row("b")], 3, {"model": "small"}
    )
    with (tmp_path / "predictions.csv").open("a", encoding="utf-8") as stream:
        stream.write("c,0.2,0.8,0.2,0.0,0.25\n")

    rows, metadata = load_inference_recovery(tmp_path, fingerprint, identifiers)

    assert [item["image_id"] for item in rows] == ["a", "b"]
    assert metadata["processed"] == 2


def test_recovery_rejects_fingerprint_or_order_mismatch(tmp_path: Path) -> None:
    fingerprint = inference_fingerprint({"model": "small"})
    save_inference_recovery(tmp_path, fingerprint, [row("a")], 2, {})
    with pytest.raises(ValueError, match="fingerprint"):
        load_inference_recovery(tmp_path, "different", ["a", "b"])
    with pytest.raises(ValueError, match="order"):
        load_inference_recovery(tmp_path, fingerprint, ["b", "a"])


def test_completed_recovery_is_reported(tmp_path: Path) -> None:
    fingerprint = inference_fingerprint({"model": "small"})
    save_inference_recovery(
        tmp_path, fingerprint, [row("a")], 1, {}, completed=True
    )
    rows, metadata = load_inference_recovery(tmp_path, fingerprint, ["a"])
    assert len(rows) == 1
    assert metadata["completed"] is True
