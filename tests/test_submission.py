import csv

import pytest

from src.submission import build_submission_rows, write_submission_atomic


def test_submission_preserves_template_columns_and_order(tmp_path) -> None:
    columns = ["image_id", "p_180"]
    template = [{"image_id": "b", "p_180": "0.5"}, {"image_id": "a", "p_180": "0.5"}]
    predictions = [{"image_id": "b", "p_final": 0.8}, {"image_id": "a", "p_final": 0.2}]

    rows = build_submission_rows(columns, template, predictions)
    path = write_submission_atomic(tmp_path / "submission.csv", columns, rows)

    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == columns
        assert list(reader) == [
            {"image_id": "b", "p_180": "0.8"},
            {"image_id": "a", "p_180": "0.2"},
        ]


@pytest.mark.parametrize(
    ("predictions", "message"),
    [
        ([{"image_id": "wrong", "p_final": 0.2}], "order"),
        ([{"image_id": "a", "p_final": float("nan")}], "finite"),
        ([{"image_id": "a", "p_final": 1.1}], r"\[0, 1\]"),
    ],
)
def test_submission_rejects_invalid_predictions(predictions, message) -> None:
    with pytest.raises(ValueError, match=message):
        build_submission_rows(
            ["image_id", "p_180"],
            [{"image_id": "a", "p_180": "0.5"}],
            predictions,
        )
