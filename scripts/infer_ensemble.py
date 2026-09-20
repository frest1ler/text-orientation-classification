"""Run component inference sequentially and create an ensemble submission."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.diagnostics import create_contact_sheets, write_json_atomic
from src.ensemble import combine_prediction_rows, load_ensemble
from src.inference_recovery import PREDICTION_COLUMNS, save_inference_recovery
from src.project_layout import ProjectLayout
from src.recovery import copy_file_atomic
from src.registry import file_sha256
from src.submission import build_submission_rows, write_submission_atomic
from src.test_data import ZipTestDataset, zip_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--name", default="best_ensemble")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--contact-sheet-count", type=int, default=16)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def _read_completed_predictions(directory: Path, total: int) -> list[dict[str, Any]]:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if not metadata.get("completed") or metadata.get("processed") != total:
        raise ValueError(f"component inference is incomplete: {directory}")
    with (directory / "predictions.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != PREDICTION_COLUMNS:
            raise ValueError("component prediction columns are incompatible")
        return list(reader)


def main() -> None:
    args = parse_args()
    layout = ProjectLayout.from_root(args.project_dir)
    manifest = load_ensemble(layout.registry, args.name)
    test_zip = layout.data / "test.zip"
    dataset = ZipTestDataset(test_zip, "test/images/", "sample_submission.csv")
    full_size = len(dataset)
    total = full_size if args.limit is None else min(args.limit, full_size)
    if total <= 0:
        raise ValueError("inference limit must be positive")
    mode = "full" if total == full_size else f"smoke_{total}"
    component_rows = []
    for component in manifest["components"]:
        selector = component["selector"]
        recovery = (
            layout.inference_recovery
            / "ensemble_components"
            / component["checkpoint_sha256"][:12]
            / mode
        )
        command = [
            sys.executable,
            "-m",
            "scripts.infer",
            "--project-dir",
            str(layout.root),
            "--model",
            selector,
            "--recovery-dir",
            str(recovery),
            "--run-name",
            f"component_{selector}_{mode}",
            "--num-workers",
            str(args.num_workers),
            "--skip-contact-sheets",
        ]
        if args.batch_size is not None:
            command += ["--batch-size", str(args.batch_size)]
        if args.limit is not None:
            command += ["--limit", str(args.limit)]
        if args.no_resume:
            command.append("--no-resume")
        subprocess.run(command, check=True)
        component_rows.append(_read_completed_predictions(recovery, total))
    weights = [float(component["weight"]) for component in manifest["components"]]
    rows = combine_prediction_rows(component_rows, weights)
    fingerprint = manifest["ensemble_sha256"] + ":" + zip_sha256(test_zip)
    recovery = layout.inference_recovery / "ensembles" / args.name / mode
    save_inference_recovery(
        recovery,
        fingerprint,
        rows,
        total,
        {
            "ensemble": args.name,
            "ensemble_sha256": manifest["ensemble_sha256"],
            "test_zip_sha256": zip_sha256(test_zip),
        },
        completed=True,
    )
    output = layout.inference_runs / f"{args.name}_{mode}_{manifest['ensemble_sha256'][:12]}"
    output.mkdir(parents=True, exist_ok=True)
    predictions = output / "predictions.csv"
    copy_file_atomic(recovery / "predictions.csv", predictions)
    submission = None
    if total == full_size:
        submission_rows = build_submission_rows(dataset.columns, dataset.template_rows, rows)
        submission = write_submission_atomic(
            output / "submission.csv", dataset.columns, submission_rows
        )
    sheets = create_contact_sheets(
        dataset, rows, output / "contact_sheets", args.contact_sheet_count
    )
    report = {
        "ensemble": manifest,
        "test_zip_sha256": file_sha256(test_zip),
        "images": total,
        "complete_test_set": total == full_size,
        "predictions": str(predictions),
        "submission": str(submission) if submission else None,
        "contact_sheets": [str(path) for path in sheets],
    }
    write_json_atomic(output / "inference_report.json", report)
    dataset.close()
    print(json.dumps({"status": "complete", "output_dir": str(output), **report}, indent=2))


if __name__ == "__main__":
    main()
