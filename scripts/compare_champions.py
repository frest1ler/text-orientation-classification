"""Validate and compare registered model champions on the locked protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.comparison import compare_champions, write_comparison
from src.project_layout import ProjectLayout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--require-model", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout = ProjectLayout.from_root(args.project_dir)
    output_dir = args.output_dir or layout.champion_evaluation
    report = compare_champions(layout.registry, tuple(args.require_model))
    paths = write_comparison(output_dir, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "winner": report["winner"],
                "models": [record["model"] for record in report["ranking"]],
                "outputs": [str(path) for path in paths],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
