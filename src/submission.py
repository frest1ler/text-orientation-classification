"""Competition submission construction and strict validation."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any


def build_submission_rows(
    template_columns: list[str],
    template_rows: list[dict[str, str]],
    prediction_rows: list[dict[str, Any]],
) -> list[dict[str, str | float]]:
    """Replace the template probability column while preserving exact order."""
    if len(template_columns) != 2:
        raise ValueError("submission template must contain exactly two columns")
    if len(template_rows) != len(prediction_rows):
        raise ValueError("submission and prediction row counts differ")
    identifier_column, probability_column = template_columns
    output: list[dict[str, str | float]] = []
    seen: set[str] = set()
    for template, prediction in zip(template_rows, prediction_rows, strict=True):
        identifier = template[identifier_column]
        if prediction.get("image_id") != identifier:
            raise ValueError("prediction order does not match sample submission")
        if identifier in seen:
            raise ValueError("submission contains duplicate image identifiers")
        seen.add(identifier)
        probability = float(prediction["p_final"])
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("submission probabilities must be finite values in [0, 1]")
        output.append(
            {identifier_column: identifier, probability_column: probability}
        )
    return output


def write_submission_atomic(
    path: str | Path,
    columns: list[str],
    rows: list[dict[str, str | float]],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
    return path
