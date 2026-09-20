import json
from pathlib import Path

import pytest

from scripts.promote_robust_champion import validate_full_robust_run
from src.config import load_config


def write_run(path: Path, profile: str, train_samples: int, validation_samples: int) -> None:
    path.mkdir()
    config = load_config("configs/baseline.yaml").to_dict()
    (path / "config.json").write_text(json.dumps(config))
    (path / "runtime.json").write_text(
        json.dumps(
            {
                "augmentation_profile": profile,
                "train_base_samples": train_samples,
                "validation_base_samples": validation_samples,
                "initial_checkpoint": {"sha256": "abc"},
            }
        )
    )


def test_only_full_robust_run_can_be_promoted(tmp_path: Path) -> None:
    run = tmp_path / "run"
    write_run(run, "robust", 50_000, 5_000)
    assert validate_full_robust_run(run)["augmentation_profile"] == "robust"


@pytest.mark.parametrize(
    ("profile", "train_samples", "validation_samples", "message"),
    [
        ("standard", 50_000, 5_000, "profile"),
        ("robust", 2_048, 5_000, "partial train"),
        ("robust", 50_000, 512, "partial validation"),
    ],
)
def test_quick_or_standard_run_is_rejected(
    tmp_path: Path,
    profile: str,
    train_samples: int,
    validation_samples: int,
    message: str,
) -> None:
    run = tmp_path / "run"
    write_run(run, profile, train_samples, validation_samples)
    with pytest.raises(ValueError, match=message):
        validate_full_robust_run(run)
