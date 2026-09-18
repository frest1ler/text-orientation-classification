"""Evaluate the offline two-orientation Tesseract baseline on synthetic validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import load_config
from src.metrics import binary_metrics
from src.ocr import orientation_probability, tesseract_confidence
from src.synthetic import SyntheticRenderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--base-samples", type=int, default=500)
    parser.add_argument("--languages", default="rus+eng")
    parser.add_argument("--confidence-scale", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ocr_baseline.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.base_samples <= 0:
        raise ValueError("base-samples must be positive")
    config = load_config(args.config)
    renderer = SyntheticRenderer(config.synthetic, "validation", config.experiment.seed)
    targets: list[int] = []
    probabilities: list[float] = []
    rows: list[dict[str, float | int]] = []
    confidence_fn = lambda image: tesseract_confidence(image, args.languages)
    for sample_id in range(args.base_samples):
        upright = renderer.render(sample_id)
        upright_probability, upright_confidence, rotated_confidence = orientation_probability(
            upright, confidence_fn, args.confidence_scale
        )
        for target, probability, direct, rotated in (
            (0, upright_probability, upright_confidence, rotated_confidence),
            (1, 1.0 - upright_probability, rotated_confidence, upright_confidence),
        ):
            targets.append(target)
            probabilities.append(probability)
            rows.append(
                {
                    "sample_id": sample_id,
                    "target": target,
                    "probability": probability,
                    "direct_confidence": direct,
                    "rotated_confidence": rotated,
                }
            )
    report = {
        "method": "tesseract_two_orientation_confidence",
        "languages": args.languages,
        "base_samples": args.base_samples,
        "confidence_scale": args.confidence_scale,
        "metrics": binary_metrics(np.asarray(targets), np.asarray(probabilities)),
        "predictions": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "predictions"}, indent=2))


if __name__ == "__main__":
    main()
