"""Numeric and visual diagnostics for unlabelled test inference."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from src.test_data import ZipTestDataset


def _summary(values: np.ndarray) -> dict[str, Any]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "q01": float(np.quantile(values, 0.01)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.5)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def build_inference_report(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    manifest: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot build an inference report without predictions")
    symmetric = np.asarray([float(row["p_symmetric"]) for row in rows])
    final = np.asarray([float(row["p_final"]) for row in rows])
    symmetry_error = np.asarray([float(row["symmetry_error"]) for row in rows])
    if not all(np.isfinite(values).all() for values in (symmetric, final, symmetry_error)):
        raise ValueError("diagnostic predictions contain non-finite values")
    elapsed = float(metadata.get("elapsed_seconds", 0.0))
    return {
        "model": manifest["model"],
        "checkpoint": manifest["checkpoint"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "validation_symmetric_metrics": manifest["metrics"]["symmetric"],
        "calibration": calibration["final_calibrator"],
        "test_zip_sha256": metadata["test_zip_sha256"],
        "images": len(rows),
        "elapsed_seconds": elapsed,
        "images_per_second": len(rows) / elapsed if elapsed > 0 else None,
        "symmetric_probability": _summary(symmetric),
        "final_probability": _summary(final),
        "symmetry_error": _summary(symmetry_error),
        "fractions": {
            "uncertain_0.45_to_0.55": float(np.mean((final >= 0.45) & (final <= 0.55))),
            "confident_at_0.10": float(np.mean((final <= 0.10) | (final >= 0.90))),
            "very_confident_at_0.01": float(np.mean((final <= 0.01) | (final >= 0.99))),
            "symmetry_error_above_0.25": float(np.mean(symmetry_error > 0.25)),
            "symmetry_error_above_0.50": float(np.mean(symmetry_error > 0.50)),
        },
    }


def write_json_atomic(path: str | Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return path


def _contact_sheet(
    dataset: ZipTestDataset,
    rows: list[dict[str, Any]],
    indices: list[int],
    destination: Path,
) -> None:
    tile_width, tile_height, label_height = 320, 120, 34
    columns = min(4, len(indices))
    rows_count = math.ceil(len(indices) / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows_count * (tile_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for position, index in enumerate(indices):
        item = dataset[index]
        image = item["image"].copy()
        image.thumbnail((tile_width - 8, tile_height - 8), Image.Resampling.LANCZOS)
        x = (position % columns) * tile_width
        y = (position // columns) * (tile_height + label_height)
        sheet.paste(image, (x + (tile_width - image.width) // 2, y + (tile_height - image.height) // 2))
        prediction = rows[index]
        label = (
            f"{prediction['image_id']} p={float(prediction['p_final']):.3f} "
            f"sym={float(prediction['symmetry_error']):.3f}"
        )
        draw.text((x + 4, y + tile_height + 4), label, fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    sheet.save(temporary, format="PNG")
    os.replace(temporary, destination)


def create_contact_sheets(
    dataset: ZipTestDataset,
    rows: list[dict[str, Any]],
    output_dir: str | Path,
    count: int = 16,
) -> list[Path]:
    """Create four deterministic sheets for qualitative, non-training review."""
    if count <= 0 or not rows:
        raise ValueError("contact sheet count and prediction rows must be positive")
    output_dir = Path(output_dir)
    count = min(count, len(rows))
    final = np.asarray([float(row["p_final"]) for row in rows])
    symmetry = np.asarray([float(row["symmetry_error"]) for row in rows])
    selections = {
        "most_confident_upright.png": np.argsort(final)[:count],
        "most_confident_upside_down.png": np.argsort(-final)[:count],
        "most_uncertain.png": np.argsort(np.abs(final - 0.5))[:count],
        "largest_symmetry_error.png": np.argsort(-symmetry)[:count],
    }
    paths = []
    for filename, indices in selections.items():
        destination = output_dir / filename
        _contact_sheet(dataset, rows, [int(index) for index in indices], destination)
        paths.append(destination)
    return paths
