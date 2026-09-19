"""Atomic, fingerprinted recovery state for ordered test inference."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


RECOVERY_VERSION = 1
PREDICTION_COLUMNS = (
    "image_id",
    "p_direct",
    "p_rotated",
    "p_symmetric",
    "symmetry_error",
    "p_final",
)


def inference_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"version": RECOVERY_VERSION, **payload},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _write_predictions_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def save_inference_recovery(
    directory: str | Path,
    fingerprint: str,
    rows: list[dict[str, Any]],
    total: int,
    metadata: dict[str, Any],
    completed: bool = False,
) -> None:
    """Commit predictions first and metadata last, both through atomic replaces."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if len(rows) > total:
        raise ValueError("processed predictions exceed total image count")
    _write_predictions_atomic(directory / "predictions.csv", rows)
    _write_json_atomic(
        directory / "metadata.json",
        {
            **metadata,
            "version": RECOVERY_VERSION,
            "fingerprint": fingerprint,
            "processed": len(rows),
            "total": total,
            "completed": completed,
        },
    )


def load_inference_recovery(
    directory: str | Path,
    expected_fingerprint: str,
    expected_image_ids: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the last committed prefix and discard any uncommitted CSV suffix."""
    directory = Path(directory)
    metadata_path = directory / "metadata.json"
    predictions_path = directory / "predictions.csv"
    if not metadata_path.exists() and not predictions_path.exists():
        return [], {"completed": False, "processed": 0, "total": len(expected_image_ids)}
    if not metadata_path.is_file() or not predictions_path.is_file():
        raise ValueError("inference recovery is incomplete")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("version") != RECOVERY_VERSION:
        raise ValueError("inference recovery version is incompatible")
    if metadata.get("fingerprint") != expected_fingerprint:
        raise ValueError("inference recovery fingerprint is incompatible")
    processed = metadata.get("processed")
    if metadata.get("total") != len(expected_image_ids):
        raise ValueError("inference recovery total image count is incompatible")
    if not isinstance(processed, int) or not 0 <= processed <= len(expected_image_ids):
        raise ValueError("inference recovery processed count is invalid")
    with predictions_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != PREDICTION_COLUMNS:
            raise ValueError("inference recovery columns are invalid")
        rows = list(reader)
    if len(rows) < processed:
        raise ValueError("inference recovery contains fewer rows than committed")
    rows = rows[:processed]
    identifiers = [row["image_id"] for row in rows]
    if identifiers != expected_image_ids[:processed]:
        raise ValueError("inference recovery image order is incompatible")
    for row in rows:
        try:
            values = [float(row[column]) for column in PREDICTION_COLUMNS[1:]]
        except ValueError as error:
            raise ValueError("inference recovery contains a non-numeric prediction") from error
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError("inference recovery predictions must be in [0, 1]")
    if metadata.get("completed") and processed != len(expected_image_ids):
        raise ValueError("completed inference recovery does not contain every image")
    return rows, metadata
