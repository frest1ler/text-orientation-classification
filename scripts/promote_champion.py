"""Promote a completed run to its per-model Drive champion when it is better."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.champions import promote_champion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--registry-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = promote_champion(args.run_dir, args.registry_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
