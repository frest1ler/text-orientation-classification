"""Run resumable symmetric inference and persist ordered technical predictions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from src.diagnostics import build_inference_report, create_contact_sheets, write_json_atomic
from src.inference import (
    infer_batch,
    load_champion,
    make_test_loader,
    resolve_inference_batch_size,
)
from src.inference_recovery import (
    inference_fingerprint,
    load_inference_recovery,
    save_inference_recovery,
)
from src.project_layout import ProjectLayout, resolve_artifact_root
from src.recovery import copy_file_atomic
from src.registry import file_sha256, select_champion
from src.reproducibility import select_device
from src.submission import build_submission_rows, write_submission_atomic
from src.test_data import ZipTestDataset
from src.test_data import zip_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_SOURCES = (
    "src/calibration.py",
    "src/diagnostics.py",
    "src/inference.py",
    "src/inference_recovery.py",
    "src/models.py",
    "src/registry.py",
    "src/submission.py",
    "src/test_data.py",
    "src/training.py",
    "src/transforms.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--artifact-source", choices=("drive", "upload"), default="drive")
    parser.add_argument("--uploaded-path", type=Path)
    parser.add_argument("--model", default="best")
    parser.add_argument("--test-zip", type=Path)
    parser.add_argument("--recovery-dir", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-every-batches", type=int, default=10)
    parser.add_argument("--run-name")
    parser.add_argument("--contact-sheet-count", type=int, default=16)
    parser.add_argument("--skip-contact-sheets", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def finalize_run(
    layout: ProjectLayout,
    bundle: Any,
    loaded: Any,
    test_zip: Path,
    recovery_dir: Path,
    rows: list[dict[str, object]],
    metadata: dict[str, object],
    full_test_size: int,
    run_name: str | None,
    contact_sheet_count: int,
    skip_contact_sheets: bool,
) -> dict[str, object]:
    mode = "full" if len(rows) == full_test_size else f"smoke_{len(rows)}"
    output_dir = layout.inference_runs / (
        run_name or f"{bundle.model}_{mode}_{bundle.manifest['checkpoint_sha256'][:12]}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = copy_file_atomic(
        recovery_dir / "predictions.csv", output_dir / "predictions.csv"
    )
    report = build_inference_report(rows, metadata, bundle.manifest, bundle.calibration)
    report.update({"mode": mode, "complete_test_set": len(rows) == full_test_size})
    report_path = write_json_atomic(output_dir / "inference_report.json", report)

    raw_dataset = ZipTestDataset(
        test_zip,
        loaded.config["data"]["image_prefix"],
        loaded.config["data"]["sample_submission_member"],
    )
    submission_path = None
    if len(rows) == full_test_size:
        submission_rows = build_submission_rows(
            raw_dataset.columns, raw_dataset.template_rows, rows
        )
        submission_path = write_submission_atomic(
            output_dir / "submission.csv", raw_dataset.columns, submission_rows
        )
    contact_sheets: list[str] = []
    if not skip_contact_sheets:
        contact_sheets = [
            str(path)
            for path in create_contact_sheets(
                raw_dataset,
                rows,
                output_dir / "contact_sheets",
                contact_sheet_count,
            )
        ]
    raw_dataset.close()
    return {
        "output_dir": str(output_dir),
        "predictions": str(predictions_path),
        "submission": str(submission_path) if submission_path else None,
        "report": str(report_path),
        "contact_sheets": contact_sheets,
    }


def main() -> None:
    args = parse_args()
    if args.save_every_batches <= 0:
        raise ValueError("save-every-batches must be positive")
    if args.contact_sheet_count <= 0:
        raise ValueError("contact-sheet-count must be positive")
    layout = ProjectLayout.from_root(args.project_dir)
    artifact_root = resolve_artifact_root(
        layout.root, args.artifact_source, args.uploaded_path
    )
    bundle = select_champion(artifact_root, args.model)
    test_zip = args.test_zip or layout.data / "test.zip"
    test_hash = zip_sha256(test_zip)
    device = select_device()
    loaded = load_champion(bundle, device)
    batch_size = resolve_inference_batch_size(loaded.config, args.batch_size)
    dataset, _ = make_test_loader(
        str(test_zip),
        loaded.config,
        batch_size,
        args.num_workers,
        device.type == "cuda",
    )
    full_test_size = len(dataset)
    total = full_test_size if args.limit is None else min(args.limit, full_test_size)
    if total <= 0:
        raise ValueError("inference limit must be positive")
    image_ids = dataset.image_ids[:total]
    fingerprint_payload = {
        "model": bundle.model,
        "checkpoint_sha256": bundle.manifest["checkpoint_sha256"],
        "calibration_sha256": bundle.manifest["calibration_sha256"],
        "test_zip_sha256": test_hash,
        "input_height": loaded.config["model"]["input_height"],
        "input_width": loaded.config["model"]["input_width"],
        "batch_size": batch_size,
        "num_workers": args.num_workers,
        "limit": total,
        "source_sha256": {
            relative: file_sha256(PROJECT_ROOT / relative)
            for relative in INFERENCE_SOURCES
        },
    }
    fingerprint = inference_fingerprint(fingerprint_payload)
    mode = "full" if total == full_test_size else f"smoke_{total}"
    recovery_dir = args.recovery_dir or layout.inference_recovery / bundle.model / mode
    if args.no_resume:
        rows: list[dict[str, object]] = []
        metadata = {"completed": False}
    else:
        rows, metadata = load_inference_recovery(
            recovery_dir, fingerprint, image_ids
        )
    if metadata.get("completed"):
        dataset.close()
        outputs = finalize_run(
            layout, bundle, loaded, test_zip, recovery_dir, rows, metadata,
            full_test_size, args.run_name, args.contact_sheet_count,
            args.skip_contact_sheets,
        )
        print(json.dumps({"status": "already_complete", **metadata, **outputs}, indent=2))
        return
    start_index = len(rows)
    if start_index == total:
        save_inference_recovery(
            recovery_dir,
            fingerprint,
            rows,
            total,
            {
                **fingerprint_payload,
                "selected_model": bundle.model,
                "device": str(device),
            },
            completed=True,
        )
        rows, metadata = load_inference_recovery(recovery_dir, fingerprint, image_ids)
        dataset.close()
        outputs = finalize_run(
            layout, bundle, loaded, test_zip, recovery_dir, rows, metadata,
            full_test_size, args.run_name, args.contact_sheet_count,
            args.skip_contact_sheets,
        )
        print(json.dumps({"status": "complete", **metadata, **outputs}, indent=2))
        return
    dataset.close()
    dataset, loader = make_test_loader(
        str(test_zip),
        loaded.config,
        batch_size,
        args.num_workers,
        device.type == "cuda",
        start_index=start_index,
        end_index=total,
    )
    common_metadata = {
        **fingerprint_payload,
        "selected_model": bundle.model,
        "device": str(device),
        "full_test_images": full_test_size,
    }
    previous_elapsed = float(metadata.get("elapsed_seconds", 0.0))
    started_at = time.perf_counter()
    progress = tqdm(loader, desc=f"inference {bundle.model}", unit="batch")
    for batch_number, batch in enumerate(progress, start=1):
        arrays = infer_batch(
            loaded.model,
            batch["image"],
            batch["rotated"],
            loaded.calibrator,
            device,
        )
        for index, image_id in enumerate(batch["image_id"]):
            if len(rows) >= total:
                break
            rows.append(
                {
                    "image_id": image_id,
                    **{name: float(values[index]) for name, values in arrays.items()},
                }
            )
        if batch_number % args.save_every_batches == 0 or len(rows) == total:
            save_inference_recovery(
                recovery_dir,
                fingerprint,
                rows,
                total,
                {
                    **common_metadata,
                    "elapsed_seconds": previous_elapsed
                    + (time.perf_counter() - started_at),
                },
                completed=len(rows) == total,
            )
    dataset.close()
    elapsed = previous_elapsed + (time.perf_counter() - started_at)
    # Ensure the final metadata contains elapsed time even when the last batch was saved above.
    save_inference_recovery(
        recovery_dir,
        fingerprint,
        rows,
        total,
        {**common_metadata, "elapsed_seconds": elapsed},
        completed=len(rows) == total,
    )
    rows, metadata = load_inference_recovery(recovery_dir, fingerprint, image_ids)
    outputs = {}
    if len(rows) == total:
        outputs = finalize_run(
            layout, bundle, loaded, test_zip, recovery_dir, rows, metadata,
            full_test_size, args.run_name, args.contact_sheet_count,
            args.skip_contact_sheets,
        )
    print(
        json.dumps(
            {
                "status": "complete" if len(rows) == total else "partial",
                "model": bundle.model,
                "processed": len(rows),
                "total": total,
                "recovery_dir": str(recovery_dir),
                **outputs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
