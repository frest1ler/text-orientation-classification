"""Fine-tune a supported standard champion on standard or robust data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from scripts.train import make_loader, write_json
from src.config import load_config
from src.models import build_model, count_parameters, unfreeze_model
from src.recovery import inspect_recovery, load_recovery, save_recovery
from src.registry import file_sha256
from src.reproducibility import make_generator, seed_everything, seed_worker, select_device
from src.robust_augmentation import ProfiledPairedSyntheticDataset
from src.synthetic import PairedSyntheticDataset
from src.training import evaluate_paired, save_checkpoint, train_paired_epoch
from src.transforms import build_preprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBUST_SOURCES = (
    "scripts/train_robust.py",
    "src/robust_augmentation.py",
    "src/training.py",
    "src/models.py",
)
SUPPORTED_ROBUST_MODELS = frozenset({"mobilenet_v3_large", "vit_b_16"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recovery-dir", type=Path, required=True)
    parser.add_argument("--augmentation-profile", choices=("standard", "robust"), required=True)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--train-base-samples", type=int)
    parser.add_argument("--validation-base-samples", type=int)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--validation-batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def robust_recovery_fingerprint(config: dict[str, Any], runtime: dict[str, Any]) -> str:
    payload = {
        "protocol_version": 1,
        "config": config,
        "runtime": runtime,
        "source_sha256": {
            name: file_sha256(PROJECT_ROOT / name) for name in ROBUST_SOURCES
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _load_initial_weights(
    model: torch.nn.Module, checkpoint_path: Path, expected_model: str
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint.get("config", {})
    checkpoint_model = config.get("model", {}).get("name")
    if checkpoint_model != expected_model:
        raise ValueError(
            "initial checkpoint model does not match robust config: "
            f"expected {expected_model!r}, got {checkpoint_model!r}"
        )
    if "model_state" not in checkpoint:
        raise ValueError("initial checkpoint does not contain model_state")
    try:
        model.load_state_dict(checkpoint["model_state"])
    except RuntimeError as error:
        raise ValueError(
            f"initial checkpoint weights are incompatible with {expected_model!r}"
        ) from error
    return {
        "path": str(checkpoint_path),
        "sha256": file_sha256(checkpoint_path),
        "epoch": checkpoint.get("epoch"),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if config.model.name not in SUPPORTED_ROBUST_MODELS:
        supported = ", ".join(sorted(SUPPORTED_ROBUST_MODELS))
        raise ValueError(
            f"robust experiment does not support {config.model.name!r}; "
            f"choose one of: {supported}"
        )
    train_samples = args.train_base_samples or config.synthetic.train_samples
    validation_samples = args.validation_base_samples or config.synthetic.validation_samples
    batch_size = args.batch_size or config.training.batch_size
    validation_batch_size = args.validation_batch_size or config.inference.batch_size
    num_workers = config.data.num_workers if args.num_workers is None else args.num_workers
    if min(train_samples, validation_samples, batch_size, validation_batch_size, args.epochs) <= 0:
        raise ValueError("sample counts, batch sizes, and epochs must be positive")
    if num_workers < 0 or args.learning_rate <= 0:
        raise ValueError("num-workers must be non-negative and learning-rate positive")

    seed_everything(config.experiment.seed, config.training.deterministic)
    device = select_device()
    preprocess = build_preprocess(
        config.model.name, config.model.input_height, config.model.input_width
    )
    train_dataset = ProfiledPairedSyntheticDataset(
        config.synthetic,
        train_samples,
        config.experiment.seed,
        args.augmentation_profile,
        preprocess,
    )
    validation_dataset = PairedSyntheticDataset(
        config.synthetic,
        "validation",
        validation_samples,
        config.experiment.seed,
        preprocess,
    )
    # Workers are recreated for every epoch so set_epoch is visible portably.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=make_generator(config.experiment.seed),
    )
    validation_loader = make_loader(
        validation_dataset,
        validation_batch_size,
        False,
        num_workers,
        config.experiment.seed,
        device.type == "cuda",
    )
    model = build_model(
        config.model.name,
        dropout=config.model.dropout,
        pretrained=args.initial_checkpoint is None,
        input_height=config.model.input_height,
        input_width=config.model.input_width,
    ).to(device)
    initial = (
        _load_initial_weights(model, args.initial_checkpoint, config.model.name)
        if args.initial_checkpoint is not None
        else {"path": None, "sha256": None, "epoch": None}
    )
    unfreeze_model(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=config.training.weight_decay
    )
    runtime = {
        "pretrained": args.initial_checkpoint is None,
        "train_base_samples": train_samples,
        "validation_base_samples": validation_samples,
        "num_workers": num_workers,
        "batch_size": batch_size,
        "validation_batch_size": validation_batch_size,
        "frozen_epochs": 0,
        "finetune_epochs": args.epochs,
        "run_name": args.output_dir.name,
        "device": str(device),
        "augmentation_profile": args.augmentation_profile,
        "epoch_varying_augmentation": args.augmentation_profile == "robust",
        "learning_rate": args.learning_rate,
        "initial_checkpoint": initial,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "config.json", config.to_dict())
    write_json(args.output_dir / "runtime.json", runtime)
    fingerprint = robust_recovery_fingerprint(config.to_dict(), runtime)
    resume_available = args.resume and (args.recovery_dir / "last.pt").is_file()
    history: list[dict[str, Any]] = []
    start_epoch = 0
    best_brier = float("inf")
    patience_used = 0
    if resume_available:
        status = inspect_recovery(args.recovery_dir, fingerprint)
        restored = load_recovery(
            args.recovery_dir,
            model,
            optimizer,
            fingerprint,
            train_loader.generator,
            map_location="cpu",
        )
        history = restored["history"]
        start_epoch = restored["epoch"]
        best_brier = restored["best_brier"]
        patience_used = restored["patience_used"]
        if (args.recovery_dir / "best.pt").is_file():
            from src.recovery import copy_file_atomic

            copy_file_atomic(args.recovery_dir / "best.pt", args.output_dir / "best.pt")
        write_json(args.output_dir / "history.json", history)
        if status["completed"]:
            print("Robust recovery is already complete; training skipped.")
            return

    print(
        json.dumps(
            {
                "device": str(device),
                "model": config.model.name,
                "parameters": count_parameters(model),
                "augmentation_profile": args.augmentation_profile,
                "initial_checkpoint": initial,
                "train_pairs": len(train_dataset),
                "validation_pairs": len(validation_dataset),
                "epochs": args.epochs,
                "resume_available": resume_available,
            },
            indent=2,
        )
    )
    completed_epoch = start_epoch
    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        losses = train_paired_epoch(
            model,
            train_loader,
            optimizer,
            device,
            config.training.symmetry_loss_weight,
            config.training.amp,
            f"{args.augmentation_profile} {epoch}/{args.epochs} train",
        )
        metrics = evaluate_paired(
            model,
            validation_loader,
            device,
            f"{args.augmentation_profile} {epoch}/{args.epochs} validation",
        )
        record = {"epoch": epoch, "phase": "robust_finetune", **losses, **metrics}
        history.append(record)
        current_brier = float(metrics["symmetric"]["brier_score"])
        if current_brier < best_brier:
            best_brier, patience_used = current_brier, 0
            save_checkpoint(
                args.output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                metrics,
                config.to_dict(),
            )
            from src.recovery import copy_file_atomic

            copy_file_atomic(args.output_dir / "best.pt", args.recovery_dir / "best.pt")
        else:
            patience_used += 1
        write_json(args.output_dir / "history.json", history)
        state = {
            "phase": "robust_finetune",
            "epoch": epoch,
            "phase_epoch": epoch,
            "best_brier": best_brier,
            "patience_used": patience_used,
            "history": history,
        }
        save_recovery(
            args.recovery_dir,
            model,
            optimizer,
            state,
            fingerprint,
            train_loader.generator,
        )
        completed_epoch = epoch
        print(json.dumps(record, sort_keys=True))
        if patience_used >= config.validation.early_stopping_patience:
            break

    save_recovery(
        args.recovery_dir,
        model,
        optimizer,
        {
            "phase": "robust_finetune",
            "epoch": completed_epoch,
            "phase_epoch": completed_epoch,
            "best_brier": best_brier,
            "patience_used": patience_used,
            "history": history,
        },
        fingerprint,
        train_loader.generator,
        completed=True,
    )


if __name__ == "__main__":
    main()
