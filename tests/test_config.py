from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.config import load_config


CONFIG_PATH = Path("configs/baseline.yaml")
EFFICIENTNET_CONFIG_PATH = Path("configs/efficientnet_b0.yaml")


def read_raw_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def write_config(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_baseline_config_is_valid() -> None:
    config = load_config(CONFIG_PATH)

    assert config.model.name == "mobilenet_v3_large"
    assert (config.model.input_height, config.model.input_width) == (96, 384)
    assert config.validation.primary_metric == "brier_score"
    assert config.to_dict()["experiment"]["seed"] == 42


def test_efficientnet_config_is_valid() -> None:
    config = load_config(EFFICIENTNET_CONFIG_PATH)

    assert config.model.name == "efficientnet_b0"
    assert config.experiment.name == "efficientnet_b0_384x96"


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    raw = deepcopy(read_raw_config())
    raw["model"]["unexpected"] = True

    with pytest.raises(ValueError, match="unknown keys"):
        load_config(write_config(tmp_path, raw))


def test_invalid_input_dimension_is_rejected(tmp_path: Path) -> None:
    raw = deepcopy(read_raw_config())
    raw["model"]["input_width"] = 385

    with pytest.raises(ValueError, match="divisible by 16"):
        load_config(write_config(tmp_path, raw))


def test_invalid_geometry_percentiles_are_rejected(tmp_path: Path) -> None:
    raw = deepcopy(read_raw_config())
    raw["synthetic"]["geometry_percentiles"][3] = 0.01

    with pytest.raises(ValueError, match="strictly increasing"):
        load_config(write_config(tmp_path, raw))
