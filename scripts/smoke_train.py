"""Run a small CPU-friendly end-to-end training and checkpoint smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.models import build_model, count_parameters
from src.reproducibility import make_generator, seed_everything, seed_worker, select_device
from src.synthetic import PairedSyntheticDataset
from src.training import evaluate_paired, load_checkpoint, save_checkpoint, train_paired_epoch
from src.transforms import ResizePadToTensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--train-base-samples", type=int, default=128)
    parser.add_argument("--validation-base-samples", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/experiments/smoke/best.pt")
    )
    return parser.parse_args()


def make_loader(dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config.experiment.seed, config.training.deterministic)
    device = select_device()
    transform = ResizePadToTensor(config.model.input_height, config.model.input_width)
    train_dataset = PairedSyntheticDataset(
        config.synthetic,
        "train",
        args.train_base_samples,
        config.experiment.seed,
        transform,
    )
    validation_dataset = PairedSyntheticDataset(
        config.synthetic,
        "validation",
        args.validation_base_samples,
        config.experiment.seed,
        transform,
    )
    train_loader = make_loader(train_dataset, args.batch_size, True, config.experiment.seed)
    validation_loader = make_loader(
        validation_dataset, args.batch_size, False, config.experiment.seed
    )

    model = build_model("small_cnn", dropout=config.model.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.frozen_learning_rate,
        weight_decay=config.training.weight_decay,
    )
    history = []
    best_brier = float("inf")
    for epoch in range(1, args.epochs + 1):
        losses = train_paired_epoch(
            model,
            train_loader,
            optimizer,
            device,
            symmetry_loss_weight=config.training.symmetry_loss_weight,
            amp=config.training.amp,
        )
        metrics = evaluate_paired(model, validation_loader, device)
        record = {"epoch": epoch, **losses, **metrics}
        history.append(record)
        print(json.dumps(record, sort_keys=True))
        if metrics["symmetric"]["brier_score"] < best_brier:
            best_brier = metrics["symmetric"]["brier_score"]
            save_checkpoint(
                args.checkpoint,
                model,
                optimizer,
                epoch,
                metrics,
                config.to_dict(),
            )

    restored = build_model("small_cnn", dropout=config.model.dropout).to(device)
    metadata = load_checkpoint(args.checkpoint, restored, map_location=device)
    restored_metrics = evaluate_paired(restored, validation_loader, device)
    result = {
        "device": str(device),
        "parameters": count_parameters(restored),
        "best_epoch": metadata["epoch"],
        "best_metrics": metadata["metrics"],
        "restored_metrics": restored_metrics,
        "history": history,
        "checkpoint": str(args.checkpoint),
    }
    result_path = args.checkpoint.with_suffix(".json")
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
