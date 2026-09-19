"""Comparable champion ranking and human-readable experiment reports."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from src.registry import select_champion


def _ranking_key(record: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = record["symmetric"]
    return (
        float(metrics["brier_score"]),
        float(metrics["log_loss"]),
        -float(metrics["roc_auc"]),
        -float(metrics["accuracy"]),
    )


def compare_champions(
    registry_dir: str | Path,
    required_models: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Validate and rank champions sharing exactly one validation protocol."""
    registry_dir = Path(registry_dir)
    leaderboard_path = registry_dir / "leaderboard.json"
    leaderboard = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    models = leaderboard.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("leaderboard does not contain model champions")
    missing = sorted(set(required_models) - set(models))
    if missing:
        raise ValueError(f"required model champions are missing: {missing}")
    protocols = {
        record.get("validation_protocol_sha256") for record in models.values()
    }
    if len(protocols) != 1:
        raise ValueError("champions use incompatible validation protocols")

    records = []
    for model_name in models:
        bundle = select_champion(registry_dir, model_name)
        manifest = bundle.manifest
        metrics = manifest["metrics"]
        records.append(
            {
                "model": model_name,
                "epoch": int(manifest["epoch"]),
                "checkpoint": manifest["checkpoint"],
                "checkpoint_mib": bundle.checkpoint_path.stat().st_size / (1024**2),
                "calibration_method": bundle.calibration["final_calibrator"]["method"],
                "symmetric": metrics["symmetric"],
                "direct": metrics.get("direct"),
                "mean_symmetry_error": metrics.get("mean_symmetry_error"),
                "runtime": manifest.get("runtime"),
                "environment": manifest.get("environment"),
            }
        )
    records.sort(key=_ranking_key)
    for rank, record in enumerate(records, start=1):
        record["rank"] = rank
        record["is_best"] = rank == 1
    return {
        "status": "comparable",
        "validation_protocol_sha256": next(iter(protocols)),
        "winner": records[0]["model"],
        "ranking": records,
    }


def _atomic_text(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)
    return path


def write_comparison(output_dir: str | Path, report: dict[str, Any]) -> list[Path]:
    """Write JSON, flat CSV, and concise Markdown from one ranking payload."""
    output_dir = Path(output_dir)
    json_path = _atomic_text(
        output_dir / "champion_comparison.json",
        json.dumps(report, indent=2),
    )
    columns = (
        "rank",
        "model",
        "brier_score",
        "log_loss",
        "roc_auc",
        "accuracy",
        "mean_symmetry_error",
        "checkpoint_mib",
        "calibration_method",
    )
    csv_path = output_dir / "champion_comparison.csv"
    temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in report["ranking"]:
            symmetric = record["symmetric"]
            writer.writerow(
                {
                    "rank": record["rank"],
                    "model": record["model"],
                    "brier_score": symmetric["brier_score"],
                    "log_loss": symmetric["log_loss"],
                    "roc_auc": symmetric["roc_auc"],
                    "accuracy": symmetric["accuracy"],
                    "mean_symmetry_error": record["mean_symmetry_error"],
                    "checkpoint_mib": f"{record['checkpoint_mib']:.2f}",
                    "calibration_method": record["calibration_method"],
                }
            )
    os.replace(temporary, csv_path)
    lines = [
        "# Champion comparison",
        "",
        f"Validation protocol: `{report['validation_protocol_sha256']}`",
        "",
        "| Rank | Model | Brier ↓ | Log loss ↓ | ROC-AUC ↑ | Accuracy ↑ |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for record in report["ranking"]:
        metrics = record["symmetric"]
        lines.append(
            f"| {record['rank']} | {record['model']} | "
            f"{float(metrics['brier_score']):.6f} | "
            f"{float(metrics['log_loss']):.6f} | "
            f"{float(metrics['roc_auc']):.6f} | "
            f"{float(metrics['accuracy']):.6f} |"
        )
    lines.extend(("", f"Selected winner: **{report['winner']}**", ""))
    markdown_path = _atomic_text(
        output_dir / "champion_comparison.md", "\n".join(lines)
    )
    return [json_path, csv_path, markdown_path]
