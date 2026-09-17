"""Typed configuration loading and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml


SUPPORTED_MODELS = {
    "small_cnn",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "vit_b_16",
}
SUPPORTED_LANGUAGES = {"ru", "en", "digits", "mixed"}


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    output_dir: str


@dataclass(frozen=True)
class DataConfig:
    test_zip: str
    image_prefix: str
    sample_submission_member: str
    num_workers: int


@dataclass(frozen=True)
class SyntheticConfig:
    train_samples: int
    validation_samples: int
    languages: list[str]
    validation_font_fraction: float


@dataclass(frozen=True)
class ModelConfig:
    name: str
    pretrained: bool
    input_height: int
    input_width: int
    dropout: float


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    frozen_epochs: int
    finetune_epochs: int
    frozen_learning_rate: float
    finetune_learning_rate: float
    weight_decay: float
    amp: bool
    deterministic: bool
    symmetry_loss_weight: float


@dataclass(frozen=True)
class ValidationConfig:
    primary_metric: str
    early_stopping_patience: int


@dataclass(frozen=True)
class InferenceConfig:
    batch_size: int
    symmetric_tta: bool
    temperature: float


@dataclass(frozen=True)
class Config:
    experiment: ExperimentConfig
    data: DataConfig
    synthetic: SyntheticConfig
    model: ModelConfig
    training: TrainingConfig
    validation: ValidationConfig
    inference: InferenceConfig

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable snapshot suitable for experiment artifacts."""
        return asdict(self)


ConfigSection = TypeVar("ConfigSection")


def _build_section(
    section_type: type[ConfigSection], values: Any, section_name: str
) -> ConfigSection:
    if not isinstance(values, dict):
        raise ValueError(f"Config section '{section_name}' must be a mapping")
    expected = {field.name for field in fields(section_type)}
    received = set(values)
    missing = sorted(expected - received)
    unknown = sorted(received - expected)
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"missing keys: {missing}")
        if unknown:
            parts.append(f"unknown keys: {unknown}")
        raise ValueError(f"Invalid config section '{section_name}': " + "; ".join(parts))
    return section_type(**values)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_config(config: Config) -> None:
    """Validate cross-field and value constraints before expensive work."""
    _require(bool(config.experiment.name.strip()), "experiment.name must not be empty")
    _require(config.experiment.seed >= 0, "experiment.seed must be non-negative")
    _require(config.data.num_workers >= 0, "data.num_workers must be non-negative")
    _require(config.data.image_prefix.endswith("/"), "data.image_prefix must end with '/'")

    _require(config.synthetic.train_samples > 0, "synthetic.train_samples must be positive")
    _require(
        config.synthetic.validation_samples > 0,
        "synthetic.validation_samples must be positive",
    )
    _require(bool(config.synthetic.languages), "synthetic.languages must not be empty")
    unknown_languages = sorted(set(config.synthetic.languages) - SUPPORTED_LANGUAGES)
    _require(not unknown_languages, f"unsupported synthetic languages: {unknown_languages}")
    _require(
        0 < config.synthetic.validation_font_fraction < 1,
        "synthetic.validation_font_fraction must be between 0 and 1",
    )

    _require(config.model.name in SUPPORTED_MODELS, f"unsupported model: {config.model.name}")
    _require(config.model.input_height > 0, "model.input_height must be positive")
    _require(config.model.input_width > 0, "model.input_width must be positive")
    _require(
        config.model.input_height % 16 == 0 and config.model.input_width % 16 == 0,
        "model input dimensions must be divisible by 16",
    )
    _require(0 <= config.model.dropout < 1, "model.dropout must be in [0, 1)")

    _require(config.training.batch_size > 0, "training.batch_size must be positive")
    _require(config.training.frozen_epochs >= 0, "training.frozen_epochs must be non-negative")
    _require(config.training.finetune_epochs > 0, "training.finetune_epochs must be positive")
    _require(
        config.training.frozen_learning_rate > 0,
        "training.frozen_learning_rate must be positive",
    )
    _require(
        config.training.finetune_learning_rate > 0,
        "training.finetune_learning_rate must be positive",
    )
    _require(config.training.weight_decay >= 0, "training.weight_decay must be non-negative")
    _require(
        config.training.symmetry_loss_weight >= 0,
        "training.symmetry_loss_weight must be non-negative",
    )

    _require(
        config.validation.primary_metric == "brier_score",
        "validation.primary_metric must be 'brier_score'",
    )
    _require(
        config.validation.early_stopping_patience > 0,
        "validation.early_stopping_patience must be positive",
    )
    _require(config.inference.batch_size > 0, "inference.batch_size must be positive")
    _require(config.inference.temperature > 0, "inference.temperature must be positive")


def load_config(path: str | Path) -> Config:
    """Load a strict YAML configuration and validate it."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")

    section_types: dict[str, type[Any]] = {
        "experiment": ExperimentConfig,
        "data": DataConfig,
        "synthetic": SyntheticConfig,
        "model": ModelConfig,
        "training": TrainingConfig,
        "validation": ValidationConfig,
        "inference": InferenceConfig,
    }
    expected_sections = set(section_types)
    received_sections = set(raw)
    if received_sections != expected_sections:
        missing = sorted(expected_sections - received_sections)
        unknown = sorted(received_sections - expected_sections)
        raise ValueError(f"Invalid config sections: missing={missing}, unknown={unknown}")

    config = Config(
        **{
            name: _build_section(section_type, raw[name], name)
            for name, section_type in section_types.items()
        }
    )
    validate_config(config)
    return config
