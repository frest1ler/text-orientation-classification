"""Generate a deterministic contact sheet of synthetic orientation pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from src.config import load_config
from src.robust_augmentation import ProfiledPairedSyntheticDataset
from src.synthetic import SyntheticOrientationDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--augmentation-profile", choices=("standard", "robust"), default="standard")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/synthetic_preview.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.augmentation_profile == "robust":
        if args.split != "train":
            raise ValueError("robust augmentation is train-only")
        paired = ProfiledPairedSyntheticDataset(
            config.synthetic,
            args.pairs,
            config.experiment.seed,
            "robust",
        )
        paired.set_epoch(args.epoch)
        samples = []
        for pair_index in range(args.pairs):
            direct, rotated, target = paired.raw_pair(pair_index)
            samples.extend(((direct, target), (rotated, 1 - target)))
    else:
        dataset = SyntheticOrientationDataset(
            config.synthetic,
            split=args.split,
            base_samples=args.pairs,
            global_seed=config.experiment.seed,
        )
        samples = [dataset[index] for index in range(len(dataset))]
    tile_width, tile_height = 420, 150
    sheet = Image.new("RGB", (tile_width * 2, tile_height * args.pairs), "#dddddd")
    draw = ImageDraw.Draw(sheet)
    for pair_index in range(args.pairs):
        for label in (0, 1):
            image, actual_label = samples[pair_index * 2 + label]
            image.thumbnail((tile_width - 16, tile_height - 28), Image.Resampling.LANCZOS)
            x0 = label * tile_width
            y0 = pair_index * tile_height
            x = x0 + (tile_width - image.width) // 2
            y = y0 + (tile_height - 24 - image.height) // 2
            sheet.paste(image, (x, y))
            draw.text((x0 + 6, y0 + tile_height - 20), f"pair={pair_index} label={actual_label}", fill="black")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, optimize=True)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
