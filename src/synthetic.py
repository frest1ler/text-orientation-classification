"""Deterministic synthetic text rendering and paired orientation dataset."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from torch.utils.data import Dataset

from src.config import SyntheticConfig
from src.text_corpus import Split, generate_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_font_assets(font_assets_dir: str | Path) -> dict[str, list[Path]]:
    """Validate bundled font files against the committed SHA-256 manifest."""
    root = Path(font_assets_dir)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    manifest_path = root / "manifest.json"
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)

    font_paths: dict[str, list[Path]] = {}
    for split in ("train", "validation"):
        entries = manifest.get(split)
        if not isinstance(entries, dict) or not entries:
            raise ValueError(f"Font manifest split '{split}' is missing or empty")
        paths = []
        for filename, expected_hash in sorted(entries.items()):
            path = root / split / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing bundled font: {path}")
            actual_hash = _sha256(path)
            if actual_hash != expected_hash:
                raise ValueError(f"Bundled font checksum mismatch: {path}")
            paths.append(path)
        font_paths[split] = paths
    if {path.name for path in font_paths["train"]} & {
        path.name for path in font_paths["validation"]
    }:
        raise ValueError("Train and validation font filenames overlap")
    return font_paths


def sample_seed(global_seed: int, split: Split, sample_id: int) -> int:
    """Derive an order-independent uint64 seed for one base sample."""
    payload = f"{global_seed}:{split}:{sample_id}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _background(rng: np.random.Generator, width: int, height: int) -> tuple[Image.Image, int]:
    base = int(rng.integers(25, 231))
    style = int(rng.integers(0, 3))
    if style == 0:
        array = np.full((height, width, 3), base, dtype=np.float32)
    elif style == 1:
        endpoint = int(np.clip(base + rng.integers(-45, 46), 10, 245))
        gradient = np.linspace(base, endpoint, width, dtype=np.float32)
        array = np.broadcast_to(gradient[None, :, None], (height, width, 3)).copy()
    else:
        noise = rng.normal(0, float(rng.uniform(2, 15)), (height, width, 1))
        array = np.full((height, width, 3), base, dtype=np.float32) + noise
    array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB"), base


def _text_color(rng: np.random.Generator, background_level: int) -> tuple[int, int, int]:
    if background_level >= 128:
        level = int(rng.integers(0, max(15, background_level - 65)))
    else:
        level = int(rng.integers(min(240, background_level + 65), 256))
    jitter = rng.integers(-20, 21, size=3)
    return tuple(int(value) for value in np.clip(level + jitter, 0, 255))


def _sample_quantiles(
    rng: np.random.Generator, percentiles: list[float], values: list[float]
) -> float:
    """Sample a continuous value from a piecewise-linear empirical CDF."""
    probability = float(rng.random())
    return float(np.interp(probability, percentiles, values))


def sample_target_size(rng: np.random.Generator, config: SyntheticConfig) -> tuple[int, int]:
    height = max(
        1,
        round(
            _sample_quantiles(
                rng, config.geometry_percentiles, config.height_quantiles
            )
        ),
    )
    aspect_ratio = _sample_quantiles(
        rng, config.geometry_percentiles, config.aspect_ratio_quantiles
    )
    width = int(np.clip(round(height * aspect_ratio), config.min_render_width, config.max_render_width))
    return width, height


def _draw_fitted_text(
    background: Image.Image,
    text: str,
    font: ImageFont.FreeTypeFont,
    color: tuple[int, int, int],
    spacing: int,
    stroke_width: int,
    rng: np.random.Generator,
) -> Image.Image:
    """Draw text through an alpha layer, fitting it without aspect distortion."""
    probe = Image.new("L", (1, 1))
    bbox = ImageDraw.Draw(probe).multiline_textbbox(
        (0, 0), text, font=font, spacing=spacing, stroke_width=stroke_width
    )
    text_width = max(1, bbox[2] - bbox[0])
    text_height = max(1, bbox[3] - bbox[1])
    layer = Image.new("L", (text_width, text_height), 0)
    ImageDraw.Draw(layer).multiline_text(
        (-bbox[0], -bbox[1]),
        text,
        font=font,
        fill=255,
        spacing=spacing,
        stroke_width=stroke_width,
    )

    margin_x = max(1, round(background.width * float(rng.uniform(0.01, 0.08))))
    margin_y = max(1, round(background.height * float(rng.uniform(0.03, 0.18))))
    available_width = max(1, background.width - 2 * margin_x)
    available_height = max(1, background.height - 2 * margin_y)
    scale = min(1.0, available_width / layer.width, available_height / layer.height)
    if scale < 1:
        layer = layer.resize(
            (
                max(1, round(layer.width * scale)),
                max(1, round(layer.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
    max_x = max(0, background.width - layer.width)
    max_y = max(0, background.height - layer.height)
    x = int(rng.integers(0, max_x + 1)) if max_x else 0
    y = int(rng.integers(0, max_y + 1)) if max_y else 0
    foreground = Image.new("RGB", background.size, color)
    mask = Image.new("L", background.size, 0)
    mask.paste(layer, (x, y))
    return Image.composite(foreground, background, mask)


def _augment(image: Image.Image, rng: np.random.Generator, jpeg_probability: float) -> Image.Image:
    if rng.random() < 0.75:
        image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.75, 1.25)))
    if rng.random() < 0.75:
        image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(0.65, 1.35)))
    blurred = rng.random() < 0.25
    if blurred:
        image = image.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.2, 1.2))))
    if rng.random() < (0.2 if blurred else 0.35):
        array = np.asarray(image, dtype=np.int16)
        noise = rng.normal(0, float(rng.uniform(1, 8)), array.shape[:2] + (1,))
        image = Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8), mode="RGB")
    if rng.random() < 0.45:
        angle = float(rng.uniform(-2.5, 2.5))
        fill = tuple(int(value) for value in np.asarray(image).reshape(-1, 3).mean(axis=0))
        image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=fill)
    if rng.random() < jpeg_probability:
        buffer = io.BytesIO()
        minimum_quality = 60 if blurred else 45
        image.save(
            buffer,
            format="JPEG",
            quality=int(rng.integers(minimum_quality, 96)),
            optimize=False,
        )
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            image = decoded.convert("RGB")
    return image


class SyntheticRenderer:
    """Render deterministic upright base images for one configured split."""

    def __init__(self, config: SyntheticConfig, split: Split, global_seed: int):
        if split not in {"train", "validation"}:
            raise ValueError(f"Unsupported split: {split}")
        self.config = config
        self.split = split
        self.global_seed = global_seed
        self.fonts = verify_font_assets(config.font_assets_dir)[split]

    def render(self, sample_id: int) -> Image.Image:
        rng = np.random.default_rng(sample_seed(self.global_seed, self.split, sample_id))
        language = self.config.languages[int(rng.integers(0, len(self.config.languages)))]
        text = generate_text(
            rng,
            split=self.split,
            language=language,
            min_words=self.config.min_words,
            max_words=self.config.max_words,
            multiline_probability=self.config.multiline_probability,
        )
        font_path = self.fonts[int(rng.integers(0, len(self.fonts)))]
        font_size = int(rng.integers(self.config.min_font_size, self.config.max_font_size + 1))
        font = ImageFont.truetype(str(font_path), size=font_size)
        if rng.random() < 0.22:
            stroke_width = 2 if font_size >= 64 and rng.random() < 0.25 else 1
        else:
            stroke_width = 0

        spacing = int(rng.integers(0, max(2, font_size // 3) + 1))
        width, height = sample_target_size(rng, self.config)
        image, background_level = _background(rng, width, height)
        color = _text_color(rng, background_level)
        image = _draw_fitted_text(
            image,
            text,
            font,
            color,
            spacing,
            stroke_width,
            rng,
        )
        return _augment(image, rng, self.config.jpeg_probability)


class SyntheticOrientationDataset(Dataset):
    """Balanced paired dataset: upright item then its exact 180-degree copy."""

    def __init__(
        self,
        config: SyntheticConfig,
        split: Split,
        base_samples: int,
        global_seed: int,
        transform: Callable[[Image.Image], object] | None = None,
    ) -> None:
        if base_samples <= 0:
            raise ValueError("base_samples must be positive")
        self.renderer = SyntheticRenderer(config, split, global_seed)
        self.base_samples = base_samples
        self.transform = transform

    def __len__(self) -> int:
        return self.base_samples * 2

    def __getitem__(self, index: int) -> tuple[object, int]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        image = self.renderer.render(index // 2)
        label = index % 2
        if label == 1:
            image = image.transpose(Image.Transpose.ROTATE_180)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
