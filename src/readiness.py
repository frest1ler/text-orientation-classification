"""Read-only delivery checks for source code and persistent project artifacts."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import torch

from src.config import load_config
from src.project_layout import ProjectLayout, resolve_artifact_root
from src.registry import select_champion
from src.test_data import ZipTestDataset, zip_sha256


REQUIRED_SOURCE_FILES = (
    "README.md",
    "requirements.txt",
    "colab_train.ipynb",
    "colab_inference.ipynb",
    "configs/baseline.yaml",
    "configs/efficientnet_b0.yaml",
    "configs/vit_b_16.yaml",
    "scripts/train.py",
    "scripts/infer.py",
)


def _check_notebook(path: Path) -> dict[str, Any]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
        raise ValueError(f"invalid notebook structure: {path}")
    code_cells = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            code_cells += 1
            ast.parse("".join(cell.get("source", [])), filename=f"{path}:cell-{index}")
    if code_cells == 0:
        raise ValueError(f"notebook contains no code cells: {path}")
    return {"path": str(path), "code_cells": code_cells}


def verify_source(source_root: str | Path) -> dict[str, Any]:
    """Validate tracked entry points, strict configs, and notebook syntax."""
    root = Path(source_root).resolve()
    missing = [name for name in REQUIRED_SOURCE_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"required delivery files are missing: {missing}")
    configs = {}
    for name in (
        "configs/baseline.yaml",
        "configs/efficientnet_b0.yaml",
        "configs/vit_b_16.yaml",
    ):
        config = load_config(root / name)
        configs[config.model.name] = name
    notebooks = [
        _check_notebook(root / "colab_train.ipynb"),
        _check_notebook(root / "colab_inference.ipynb"),
    ]
    return {
        "status": "ready",
        "source_root": str(root),
        "required_files": len(REQUIRED_SOURCE_FILES),
        "configs": configs,
        "notebooks": notebooks,
    }


def verify_project(
    project_root: str | Path,
    artifact_source: str = "drive",
    uploaded_path: str | Path | None = None,
    model: str = "best",
) -> dict[str, Any]:
    """Validate a Drive/local test ZIP and every registered champion bundle."""
    layout = ProjectLayout.from_root(project_root)
    test_zip = layout.data / "test.zip"
    if not test_zip.is_file():
        raise FileNotFoundError(f"test archive is missing: {test_zip}")
    artifact_root = resolve_artifact_root(
        layout.root, artifact_source, uploaded_path
    )
    champions = {}
    if artifact_source == "drive":
        leaderboard_path = artifact_root / "leaderboard.json"
        if not leaderboard_path.is_file():
            raise FileNotFoundError(f"leaderboard is missing: {leaderboard_path}")
        leaderboard = json.loads(leaderboard_path.read_text(encoding="utf-8"))
        models = leaderboard.get("models")
        if not isinstance(models, dict) or not models:
            raise ValueError("leaderboard does not contain model champions")
        names = sorted(models)
    else:
        names = [select_champion(artifact_root, model).model]
    for model_name in names:
        bundle = select_champion(artifact_root, model_name)
        champions[model_name] = {
            "checkpoint": bundle.manifest["checkpoint"],
            "checkpoint_sha256": bundle.manifest["checkpoint_sha256"],
            "symmetric_brier_score": float(
                bundle.manifest["metrics"]["symmetric"]["brier_score"]
            ),
        }
    selected = select_champion(artifact_root, model)
    manifest = selected.manifest
    # Template/image agreement is checked without extracting the archive.
    checkpoint = torch.load(
        selected.checkpoint_path, map_location="cpu", weights_only=True
    )
    if not isinstance(checkpoint.get("config"), dict):
        raise ValueError("best champion checkpoint does not contain its config")
    data_config = checkpoint["config"]["data"]
    dataset = ZipTestDataset(
        test_zip,
        data_config["image_prefix"],
        data_config["sample_submission_member"],
    )
    test_images = len(dataset)
    dataset.close()
    return {
        "status": "ready",
        "project_root": str(layout.root),
        "test_zip_sha256": zip_sha256(test_zip),
        "test_images": test_images,
        "champions": champions,
        "selected_model": manifest["model"],
        "artifact_source": artifact_source,
    }
