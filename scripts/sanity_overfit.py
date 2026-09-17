"""Verify that the paired pipeline can memorize a tiny fixed dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.models import build_model
from src.reproducibility import make_generator, seed_everything, select_device
from src.synthetic import PairedSyntheticDataset
from src.training import evaluate_paired, train_paired_epoch
from src.transforms import ResizePadToTensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--base-samples", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--target-brier", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config.experiment.seed, config.training.deterministic)
    device = select_device()
    transform = ResizePadToTensor(config.model.input_height, config.model.input_width)
    dataset = PairedSyntheticDataset(
        config.synthetic,
        "train",
        args.base_samples,
        config.experiment.seed,
        transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.base_samples,
        shuffle=True,
        num_workers=0,
        generator=make_generator(config.experiment.seed),
    )
    model = build_model("small_cnn", dropout=0.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)

    final_metrics = None
    for epoch in range(1, args.epochs + 1):
        losses = train_paired_epoch(
            model,
            loader,
            optimizer,
            device,
            symmetry_loss_weight=0.1,
            amp=False,
        )
        final_metrics = evaluate_paired(model, loader, device)
        brier = final_metrics["symmetric"]["brier_score"]
        accuracy = final_metrics["symmetric"]["accuracy"]
        symmetry_error = final_metrics["mean_symmetry_error"]
        passed = brier <= args.target_brier and accuracy == 1.0 and symmetry_error <= 0.05
        if epoch == 1 or epoch % 10 == 0 or passed:
            print(json.dumps({"epoch": epoch, **losses, **final_metrics}, sort_keys=True))
        if passed:
            print(f"Sanity overfit passed at epoch {epoch}")
            return
    raise RuntimeError(
        f"Sanity overfit failed after {args.epochs} epochs: {final_metrics}"
    )


if __name__ == "__main__":
    main()
