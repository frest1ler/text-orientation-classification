"""Offline Tesseract confidence baseline for 0° versus 180° orientation."""

from __future__ import annotations

import csv
import io
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image


def parse_tesseract_confidences(tsv: str) -> float:
    """Return character-weighted mean confidence for recognised words."""
    rows = csv.DictReader(io.StringIO(tsv), delimiter="\t")
    weighted_sum = 0.0
    character_count = 0
    for row in rows:
        text = (row.get("text") or "").strip()
        try:
            confidence = float(row.get("conf", "-1"))
        except ValueError:
            continue
        if text and confidence >= 0:
            weight = len(text)
            weighted_sum += confidence * weight
            character_count += weight
    return weighted_sum / character_count if character_count else 0.0


def tesseract_confidence(image: Image.Image, languages: str = "rus+eng") -> float:
    executable = shutil.which("tesseract")
    if executable is None:
        raise RuntimeError(
            "Tesseract is not installed. Install tesseract-ocr and the required language packs."
        )
    with tempfile.TemporaryDirectory(prefix="orientation-ocr-") as directory:
        image_path = Path(directory) / "image.png"
        image.convert("RGB").save(image_path)
        process = subprocess.run(
            [executable, str(image_path), "stdout", "-l", languages, "tsv"],
            check=True,
            capture_output=True,
            text=True,
        )
    return parse_tesseract_confidences(process.stdout)


def orientation_probability(
    image: Image.Image,
    confidence_fn: Callable[[Image.Image], float] = tesseract_confidence,
    confidence_scale: float = 10.0,
) -> tuple[float, float, float]:
    """Return P(upside-down) and OCR confidence for both candidate orientations."""
    if confidence_scale <= 0:
        raise ValueError("confidence_scale must be positive")
    direct = float(confidence_fn(image))
    rotated = float(confidence_fn(image.transpose(Image.Transpose.ROTATE_180)))
    difference = np.clip((rotated - direct) / confidence_scale, -50.0, 50.0)
    probability = float(1.0 / (1.0 + np.exp(-difference)))
    return probability, direct, rotated
