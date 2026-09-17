"""Train a paired compact CNN in frozen-head and full fine-tuning phases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.config import Config, load_config
from src.models import build_model, count_parameters, freeze_backbone, unfreeze_model
from src.reproducibility import make_generator, seed_everything, seed_worker, select_device
from src.synthetic import PairedSyntheticDataset
from src.training import evaluate_paired, save_checkpoint, train_paired_epoch
from src.transforms import build_preprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--train-base-samples", type=int)
    parser.add_argument("--validation-base-samples", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--validation-batch-size", type=int)
    parser.add_argument("--frozen-epochs", type=int)
    parser.add_argument("--finetune-epochs", type=int)
    parser.add_argument("--run-name")
    return parser.parse_args()


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_phase(
    phase: str,
    epochs: int,
    start_epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    config: Config,
    output_dir: Path,
    history: list[dict[str, Any]],
    best_brier: float,
    patience_used: int,
) -> tuple[int, float, int, bool]:
    """Run one training phase and return updated experiment state."""
    epoch_number = start_epoch
    stopped_early = False
    for _ in range(epochs):
        epoch_number += 1
        losses = train_paired_epoch(
            model,
            train_loader,
            optimizer,
            device,
            symmetry_loss_weight=config.training.symmetry_loss_weight,
            amp=config.training.amp,
        )
        metrics = evaluate_paired(model, validation_loader, device)
        record = {"epoch": epoch_number, "phase": phase, **losses, **metrics}
        history.append(record)
        print(json.dumps(record, sort_keys=True))
        current_brier = metrics["symmetric"]["brier_score"]
        if current_brier < best_brier:
            best_brier = current_brier
            patience_used = 0
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch_number,
                metrics,
                config.to_dict(),
            )
        else:
            patience_used += 1
        write_json(output_dir / "history.json", history)
        if patience_used >= config.validation.early_stopping_patience:
            stopped_early = True
            break
    return epoch_number, best_brier, patience_used, stopped_early


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if config.model.name not in {"mobilenet_v3_large", "efficientnet_b0"}:
        raise ValueError("scripts/train.py currently supports compact pretrained CNNs only")
    seed_everything(config.experiment.seed, config.training.deterministic)
    device = select_device()
    preprocess = build_preprocess(
        config.model.name, config.model.input_height, config.model.input_width
    )
    train_samples = (
        config.synthetic.train_samples
        if args.train_base_samples is None
        else args.train_base_samples
    )
    validation_samples = (
        config.synthetic.validation_samples
        if args.validation_base_samples is None
        else args.validation_base_samples
    )
    num_workers = config.data.num_workers if args.num_workers is None else args.num_workers
    batch_size = config.training.batch_size if args.batch_size is None else args.batch_size
    validation_batch_size = (
        config.inference.batch_size
        if args.validation_batch_size is None
        else args.validation_batch_size
    )
    frozen_epochs = (
        config.training.frozen_epochs
        if args.frozen_epochs is None
        else args.frozen_epochs
    )
    finetune_epochs = (
        config.training.finetune_epochs
        if args.finetune_epochs is None
        else args.finetune_epochs
    )
    if train_samples <= 0 or validation_samples <= 0:
        raise ValueError("train and validation sample counts must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if batch_size <= 0 or validation_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if frozen_epochs < 0 or finetune_epochs <= 0:
        raise ValueError("frozen epochs must be non-negative and finetune epochs positive")
    train_dataset = PairedSyntheticDataset(
        config.synthetic, "train", train_samples, config.experiment.seed, preprocess
    )
    validation_dataset = PairedSyntheticDataset(
        config.synthetic,
        "validation",
        validation_samples,
        config.experiment.seed,
        preprocess,
    )
    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        train_dataset,
        batch_size,
        True,
        num_workers,
        config.experiment.seed,
        pin_memory,
    )
    validation_loader = make_loader(
        validation_dataset,
        validation_batch_size,
        False,
        num_workers,
        config.experiment.seed,
        pin_memory,
    )
    pretrained = config.model.pretrained and not args.no_pretrained
    model = build_model(
        config.model.name, dropout=config.model.dropout, pretrained=pretrained
    ).to(device)
    run_name = args.run_name or config.experiment.name
    output_dir = Path(config.experiment.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", config.to_dict())
    write_json(
        output_dir / "runtime.json",
        {
            "pretrained": pretrained,
            "train_base_samples": train_samples,
            "validation_base_samples": validation_samples,
            "num_workers": num_workers,
            "batch_size": batch_size,
            "validation_batch_size": validation_batch_size,
            "frozen_epochs": frozen_epochs,
            "finetune_epochs": finetune_epochs,
            "run_name": run_name,
            "device": str(device),
        },
    )
    print(
        json.dumps(
            {
                "device": str(device),
                "model": config.model.name,
                "pretrained": pretrained,
                "parameters": count_parameters(model),
                "train_pairs": len(train_dataset),
                "validation_pairs": len(validation_dataset),
                "num_workers": num_workers,
                "batch_size": batch_size,
                "validation_batch_size": validation_batch_size,
                "frozen_epochs": frozen_epochs,
                "finetune_epochs": finetune_epochs,
                "run_name": run_name,
            },
            indent=2,
        )
    )

    history: list[dict[str, Any]] = []
    epoch = 0
    best_brier = float("inf")
    patience_used = 0
    if frozen_epochs > 0:
        freeze_backbone(model)
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=config.training.frozen_learning_rate,
            weight_decay=config.training.weight_decay,
        )
        epoch, best_brier, patience_used, stopped = run_phase(
            "frozen",
            frozen_epochs,
            epoch,
            model,
            optimizer,
            train_loader,
            validation_loader,
            device,
            config,
            output_dir,
            history,
            best_brier,
            patience_used,
        )
        if stopped:
            print("Early stopping during frozen phase")

    unfreeze_model(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.finetune_learning_rate,
        weight_decay=config.training.weight_decay,
    )
    patience_used = 0
    run_phase(
        "finetune",
        finetune_epochs,
        epoch,
        model,
        optimizer,
        train_loader,
        validation_loader,
        device,
        config,
        output_dir,
        history,
        best_brier,
        patience_used,
    )


if __name__ == "__main__":
    main()
