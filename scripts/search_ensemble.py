"""Search validation weights and optionally promote an isolated ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ensemble import (
    build_ensemble_manifest,
    load_validation_candidate,
    promote_ensemble,
    search_ensemble,
)
from src.diagnostics import write_json_atomic
from src.project_layout import ProjectLayout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate", action="append", required=True, metavar="SELECTOR=RUN_DIR"
    )
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--no-promote", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout = ProjectLayout.from_root(args.project_dir)
    specifications = []
    for value in args.candidate:
        selector, separator, run_dir = value.partition("=")
        if not separator or not selector or not run_dir:
            raise ValueError("candidate must use SELECTOR=RUN_DIR")
        specifications.append((selector, Path(run_dir)))
    candidates = [
        load_validation_candidate(layout.registry, selector, run_dir)
        for selector, run_dir in specifications
    ]
    search = search_ensemble(candidates, args.weight_step)
    manifest = build_ensemble_manifest(candidates, search)
    report_path = layout.root / "evaluation/ensembles/search.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report_path, manifest)
    promotion = None if args.no_promote else promote_ensemble(manifest, layout.registry)
    print(json.dumps({"report": str(report_path), "manifest": manifest, "promotion": promotion}, indent=2))


if __name__ == "__main__":
    main()
