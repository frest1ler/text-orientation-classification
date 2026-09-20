"""Write the guarded standard-versus-robust validation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.project_layout import ProjectLayout
from src.robust_selection import compare_robust
from src.tuning import write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--minimum-improvement", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout = ProjectLayout.from_root(args.project_dir)
    report = compare_robust(
        layout.registry, minimum_improvement=args.minimum_improvement
    )
    destination = layout.root / "evaluation" / "robust" / "comparison.json"
    write_json_atomic(destination, report)
    print(json.dumps({**report, "report": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
