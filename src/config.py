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
    font_assets_dir: str
    min_font_size: int
    max_font_size: int
    min_words: int
    max_words: int
    multiline_probability: float
    jpeg_probability: float
    geometry_percentiles: list[float]
    height_quantiles: list[float]
    aspect_ratio_quantiles: list[float]
    min_render_width: int
    max_render_width: int


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
    _require(config.synthetic.min_font_size > 0, "synthetic.min_font_size must be positive")
    _require(
        config.synthetic.max_font_size >= config.synthetic.min_font_size,
        "synthetic.max_font_size must be >= synthetic.min_font_size",
    )
    _require(config.synthetic.min_words > 0, "synthetic.min_words must be positive")
    _require(
        config.synthetic.max_words >= config.synthetic.min_words,
        "synthetic.max_words must be >= synthetic.min_words",
    )
    _require(
        0 <= config.synthetic.multiline_probability <= 1,
        "synthetic.multiline_probability must be in [0, 1]",
    )
    _require(
        0 <= config.synthetic.jpeg_probability <= 1,
        "synthetic.jpeg_probability must be in [0, 1]",
    )
    _require(
        len(config.synthetic.geometry_percentiles) >= 2,
        "synthetic geometry requires at least two percentile points",
    )
    _require(
        len(config.synthetic.geometry_percentiles)
        == len(config.synthetic.height_quantiles)
        == len(config.synthetic.aspect_ratio_quantiles),
        "synthetic geometry quantile lists must have equal lengths",
    )
    _require(
        config.synthetic.geometry_percentiles[0] == 0
        and config.synthetic.geometry_percentiles[-1] == 1,
        "synthetic.geometry_percentiles must start at 0 and end at 1",
    )
    _require(
        all(
            left < right
            for left, right in zip(
                config.synthetic.geometry_percentiles[:-1],
                config.synthetic.geometry_percentiles[1:],
                strict=True,
            )
        ),
        "synthetic.geometry_percentiles must be strictly increasing",
    )
    _require(
        all(value > 0 for value in config.synthetic.height_quantiles),
        "synthetic.height_quantiles must be positive",
    )
    _require(
        all(value > 0 for value in config.synthetic.aspect_ratio_quantiles),
        "synthetic.aspect_ratio_quantiles must be positive",
    )
    _require(
        config.synthetic.min_render_width > 0,
        "synthetic.min_render_width must be positive",
    )
    _require(
        config.synthetic.max_render_width > 0,
        "synthetic.max_render_width must be positive",
    )
    _require(
        config.synthetic.max_render_width >= config.synthetic.min_render_width,
        "synthetic.max_render_width must be >= synthetic.min_render_width",
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
