"""Validate that source and optional persistent artifacts are ready for review."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.readiness import verify_project, verify_source


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path)
    parser.add_argument("--artifact-source", choices=("drive", "upload"), default="drive")
    parser.add_argument("--uploaded-path", type=Path)
    parser.add_argument("--model", default="best")
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    result = {"source": verify_source(SOURCE_ROOT)}
    if args.project_dir is not None:
        result["project"] = verify_project(
            args.project_dir, args.artifact_source, args.uploaded_path, args.model
        )
    if args.report is not None:
        _write_atomic(args.report, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
