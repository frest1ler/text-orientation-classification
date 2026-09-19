"""Run resumable symmetric inference and persist ordered technical predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm.auto import tqdm

from src.inference import infer_batch, load_champion, make_test_loader
from src.inference_recovery import (
    inference_fingerprint,
    load_inference_recovery,
    save_inference_recovery,
)
from src.project_layout import ProjectLayout, resolve_artifact_root
from src.registry import file_sha256, select_champion
from src.reproducibility import select_device
from src.test_data import zip_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_SOURCES = (
    "src/calibration.py",
    "src/inference.py",
    "src/inference_recovery.py",
    "src/models.py",
    "src/registry.py",
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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-every-batches", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.save_every_batches <= 0:
        raise ValueError("save-every-batches must be positive")
    layout = ProjectLayout.from_root(args.project_dir)
    artifact_root = resolve_artifact_root(
        layout.root, args.artifact_source, args.uploaded_path
    )
    bundle = select_champion(artifact_root, args.model)
    test_zip = args.test_zip or layout.data / "test.zip"
    test_hash = zip_sha256(test_zip)
    device = select_device()
    loaded = load_champion(bundle, device)
    dataset, _ = make_test_loader(
        str(test_zip),
        loaded.config,
        args.batch_size,
        args.num_workers,
        device.type == "cuda",
    )
    total = len(dataset) if args.limit is None else min(args.limit, len(dataset))
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
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "limit": total,
        "source_sha256": {
            relative: file_sha256(PROJECT_ROOT / relative)
            for relative in INFERENCE_SOURCES
        },
    }
    fingerprint = inference_fingerprint(fingerprint_payload)
    recovery_dir = args.recovery_dir or layout.inference_recovery / bundle.model
    if args.no_resume:
        rows: list[dict[str, object]] = []
        metadata = {"completed": False}
    else:
        rows, metadata = load_inference_recovery(
            recovery_dir, fingerprint, image_ids
        )
    if metadata.get("completed"):
        print(json.dumps({"status": "already_complete", **metadata}, indent=2))
        dataset.close()
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
        dataset.close()
        print(json.dumps({"status": "complete", "processed": total, "total": total}, indent=2))
        return
    dataset.close()
    dataset, loader = make_test_loader(
        str(test_zip),
        loaded.config,
        args.batch_size,
        args.num_workers,
        device.type == "cuda",
        start_index=start_index,
        end_index=total,
    )
    common_metadata = {
        **fingerprint_payload,
        "selected_model": bundle.model,
        "device": str(device),
    }
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
                common_metadata,
                completed=len(rows) == total,
            )
    dataset.close()
    print(
        json.dumps(
            {
                "status": "complete" if len(rows) == total else "partial",
                "model": bundle.model,
                "processed": len(rows),
                "total": total,
                "recovery_dir": str(recovery_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
