"""Import a completed run directory or ZIP into the unified project registry."""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from contextlib import nullcontext
from pathlib import Path, PurePosixPath

from src.champions import promote_champion
from src.project_layout import ProjectLayout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed run directory or ZIP")
    parser.add_argument("--project-dir", type=Path, required=True)
    return parser.parse_args()


def _validate_members(archive: zipfile.ZipFile) -> None:
    for name in archive.namelist():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe path in run archive: {name}")


def main() -> None:
    args = parse_args()
    layout = ProjectLayout.from_root(args.project_dir)
    layout.create_output_directories()
    context = (
        tempfile.TemporaryDirectory(prefix="champion-import-")
        if args.run.is_file()
        else nullcontext(None)
    )
    with context as temporary:
        if temporary is None:
            run_dir = args.run
        else:
            with zipfile.ZipFile(args.run) as archive:
                _validate_members(archive)
                archive.extractall(temporary)
            run_dir = Path(temporary)
        result = promote_champion(run_dir, layout.registry)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
