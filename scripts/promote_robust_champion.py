"""Promote only a full robust run into the registry's isolated robust branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.champions import promote_champion
from src.project_layout import ProjectLayout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    return parser.parse_args()


def validate_full_robust_run(run_dir: Path) -> dict:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    runtime = json.loads((run_dir / "runtime.json").read_text(encoding="utf-8"))
    if runtime.get("augmentation_profile") != "robust":
        raise ValueError("only augmentation_profile='robust' can enter robust registry")
    if runtime.get("train_base_samples") != config["synthetic"]["train_samples"]:
        raise ValueError("quick or partial train run cannot become a robust champion")
    if runtime.get("validation_base_samples") != config["synthetic"]["validation_samples"]:
        raise ValueError("quick or partial validation cannot become a robust champion")
    initial = runtime.get("initial_checkpoint")
    if not isinstance(initial, dict) or not initial.get("sha256"):
        raise ValueError("robust run must record its initial checkpoint provenance")
    return runtime


def main() -> None:
    args = parse_args()
    validate_full_robust_run(args.run_dir)
    robust_registry = ProjectLayout.from_root(args.project_dir).robust_registry
    result = promote_champion(args.run_dir, robust_registry)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
