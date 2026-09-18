"""Evaluate OOF calibration for a completed CNN run and save its calibrator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.train import make_loader
from src.calibration import evaluate_oof_calibration
from src.config import load_config
from src.models import build_model
from src.reproducibility import seed_everything, select_device
from src.synthetic import PairedSyntheticDataset
from src.training import predict_paired
from src.transforms import build_preprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = json.loads((args.run_dir / "runtime.json").read_text(encoding="utf-8"))
    config = load_config(args.config)
    seed_everything(config.experiment.seed, config.training.deterministic)
    device = select_device()
    preprocess = build_preprocess(
        config.model.name, config.model.input_height, config.model.input_width
    )
    dataset = PairedSyntheticDataset(
        config.synthetic,
        "validation",
        runtime["validation_base_samples"],
        config.experiment.seed,
        preprocess,
    )
    loader = make_loader(
        dataset,
        runtime["validation_batch_size"],
        False,
        runtime["num_workers"],
        config.experiment.seed,
        device.type == "cuda",
    )
    checkpoint = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=True)
    if checkpoint["config"] != config.to_dict():
        raise ValueError("checkpoint config does not match --config")
    model = build_model(config.model.name, dropout=config.model.dropout, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    predictions = predict_paired(model, loader, device)
    calibrator, report, oof = evaluate_oof_calibration(
        predictions["targets"],
        predictions["symmetric"],
        folds=args.folds,
        seed=config.experiment.seed,
    )
    report["checkpoint_epoch"] = checkpoint["epoch"]
    report["prediction_mode"] = "symmetric"
    (args.run_dir / "calibration.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.run_dir / "validation_predictions.npz",
        targets=predictions["targets"],
        direct=predictions["direct"],
        symmetric=predictions["symmetric"],
        **{f"oof_{name}": values for name, values in oof.items()},
    )
    print(json.dumps({**report, "final_calibrator": calibrator.to_dict()}, indent=2))


if __name__ == "__main__":
    main()
