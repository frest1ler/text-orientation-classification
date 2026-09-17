"""Validate a project config and print the resolved runtime settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config
from src.reproducibility import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    print(
        json.dumps(
            {
                "config": str(args.config),
                "experiment": config.experiment.name,
                "model": config.model.name,
                "input_size": [config.model.input_height, config.model.input_width],
                "device": str(select_device()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
