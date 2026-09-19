"""Canonical persistent directory layout shared by training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectLayout:
    root: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "ProjectLayout":
        return cls(Path(root).expanduser())

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def registry(self) -> Path:
        return self.root / "registry"

    @property
    def champions(self) -> Path:
        return self.registry / "champions"

    @property
    def training_runs(self) -> Path:
        return self.root / "training" / "runs"

    @property
    def training_recovery(self) -> Path:
        return self.root / "training" / "recovery"

    @property
    def inference_runs(self) -> Path:
        return self.root / "inference" / "runs"

    @property
    def inference_recovery(self) -> Path:
        return self.root / "inference" / "recovery"

    @property
    def ocr_runs(self) -> Path:
        return self.root / "ocr" / "runs"

    def create_output_directories(self) -> None:
        for path in (
            self.registry,
            self.training_runs,
            self.training_recovery,
            self.inference_runs,
            self.inference_recovery,
            self.ocr_runs,
        ):
            path.mkdir(parents=True, exist_ok=True)


def resolve_artifact_root(
    project_dir: str | Path,
    source: str,
    uploaded_path: str | Path | None = None,
) -> Path:
    """Resolve a registry or uploaded bundle without Colab-specific imports."""
    if source == "drive":
        if uploaded_path is not None:
            raise ValueError("uploaded_path must be omitted when source='drive'")
        return ProjectLayout.from_root(project_dir).registry
    if source == "upload":
        if uploaded_path is None:
            raise ValueError("uploaded_path is required when source='upload'")
        return Path(uploaded_path).expanduser()
    raise ValueError("artifact source must be 'drive' or 'upload'")
