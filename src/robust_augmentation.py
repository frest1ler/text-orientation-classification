"""Optional reproducible train-only degradations for robust orientation models."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import Dataset

from src.config import SyntheticConfig
from src.synthetic import SyntheticRenderer


AugmentationProfile = Literal["standard", "robust"]


def augmentation_seed(global_seed: int, epoch: int, sample_id: int) -> int:
    """Derive an order- and worker-independent seed for one train sample."""
    if epoch < 0 or sample_id < 0:
        raise ValueError("epoch and sample_id must be non-negative")
    payload = f"robust-v1:{global_seed}:{epoch}:{sample_id}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _rotation(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    angle = float(rng.uniform(-7.0, 7.0))
    fill = tuple(int(v) for v in np.asarray(image).reshape(-1, 3).mean(axis=0))
    return image.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=fill,
    )


def _perspective(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    width, height = image.size
    dx = max(1, round(width * float(rng.uniform(0.02, 0.10))))
    dy = max(1, round(height * float(rng.uniform(0.02, 0.10))))
    quad = (
        int(rng.integers(0, dx + 1)),
        int(rng.integers(0, dy + 1)),
        int(rng.integers(0, dx + 1)),
        height - int(rng.integers(0, dy + 1)),
        width - int(rng.integers(0, dx + 1)),
        height - int(rng.integers(0, dy + 1)),
        width - int(rng.integers(0, dx + 1)),
        int(rng.integers(0, dy + 1)),
    )
    fill = tuple(int(v) for v in np.asarray(image).reshape(-1, 3).mean(axis=0))
    return image.transform(
        image.size,
        Image.Transform.QUAD,
        quad,
        resample=Image.Resampling.BICUBIC,
        fillcolor=fill,
    )


def _edge_crop(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    width, height = image.size
    max_x = max(1, round(width * 0.12))
    max_y = max(1, round(height * 0.12))
    left = int(rng.integers(0, max_x + 1)) if rng.random() < 0.5 else 0
    right = int(rng.integers(0, max_x + 1)) if rng.random() < 0.5 else 0
    top = int(rng.integers(0, max_y + 1)) if rng.random() < 0.5 else 0
    bottom = int(rng.integers(0, max_y + 1)) if rng.random() < 0.5 else 0
    if left + right == 0 and top + bottom == 0:
        left = max(1, max_x // 2)
    cropped = image.crop((left, top, width - right, height - bottom))
    return cropped.resize(image.size, Image.Resampling.BICUBIC)


def _motion_blur(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    # Pillow's built-in Kernel filter accepts only 3x3 and 5x5 kernels.
    size = int(rng.choice((3, 5)))
    kernel = np.zeros((size, size), dtype=np.float32)
    direction = int(rng.integers(0, 4))
    if direction == 0:
        kernel[size // 2, :] = 1
    elif direction == 1:
        kernel[:, size // 2] = 1
    elif direction == 2:
        np.fill_diagonal(kernel, 1)
    else:
        np.fill_diagonal(np.fliplr(kernel), 1)
    return image.filter(
        ImageFilter.Kernel((size, size), kernel.flatten().tolist(), scale=float(size))
    )


def _downscale_blur(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    scale = float(rng.uniform(0.4, 0.8))
    reduced = image.resize(
        (max(2, round(image.width * scale)), max(2, round(image.height * scale))),
        Image.Resampling.BILINEAR,
    )
    return reduced.resize(image.size, Image.Resampling.BILINEAR)


def _spatial_blur(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    blurred = image.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.8, 1.8))))
    horizontal = np.linspace(0, 255, image.width, dtype=np.uint8)
    if rng.random() < 0.5:
        horizontal = horizontal[::-1]
    mask = Image.fromarray(np.broadcast_to(horizontal, (image.height, image.width)), mode="L")
    return Image.composite(blurred, image, mask)


def _shadow_or_glare(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    array = np.asarray(image, dtype=np.float32)
    axis = 1 if rng.random() < 0.5 else 0
    length = image.width if axis == 1 else image.height
    center = float(rng.uniform(0.15, 0.85) * length)
    spread = float(rng.uniform(0.18, 0.45) * length)
    coordinate = np.arange(length, dtype=np.float32)
    band = np.exp(-0.5 * ((coordinate - center) / spread) ** 2)
    shape = (1, length, 1) if axis == 1 else (length, 1, 1)
    strength = float(rng.uniform(25, 70)) * (1 if rng.random() < 0.5 else -1)
    array = np.clip(array + band.reshape(shape) * strength, 0, 255)
    return Image.fromarray(array.astype(np.uint8), mode="RGB")


def _occlusion(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    result = image.copy()
    width = max(1, round(image.width * float(rng.uniform(0.03, 0.10))))
    height = max(1, round(image.height * float(rng.uniform(0.08, 0.22))))
    x = int(rng.integers(0, max(1, image.width - width + 1)))
    y = int(rng.integers(0, max(1, image.height - height + 1)))
    pixels = np.asarray(image)
    color = tuple(int(v) for v in pixels.reshape(-1, 3).mean(axis=0))
    ImageDraw.Draw(result).rectangle((x, y, x + width, y + height), fill=color)
    return result


def apply_robust_augmentation(image: Image.Image, seed: int) -> Image.Image:
    """Apply a bounded combination of realistic train-only degradations."""
    rng = np.random.default_rng(seed)
    image = image.convert("RGB")
    operations: list[tuple[float, Callable[[Image.Image, np.random.Generator], Image.Image]]] = [
        (0.20, _rotation),
        (0.20, _perspective),
        (0.12, _edge_crop),
        (0.08, _motion_blur),
        (0.12, _downscale_blur),
        (0.05, _spatial_blur),
        (0.12, _shadow_or_glare),
        (0.05, _occlusion),
    ]
    selected = [operation for probability, operation in operations if rng.random() < probability]
    if len(selected) > 2:
        chosen = rng.choice(len(selected), size=2, replace=False)
        selected = [selected[int(index)] for index in chosen]
    for operation in selected:
        image = operation(image, rng)
    return image.convert("RGB")


class ProfiledPairedSyntheticDataset(Dataset):
    """Paired train data with optional deterministic epoch-varying degradation."""

    def __init__(
        self,
        config: SyntheticConfig,
        base_samples: int,
        global_seed: int,
        profile: AugmentationProfile,
        transform: Callable[[Image.Image], object] | None = None,
    ) -> None:
        if base_samples <= 0:
            raise ValueError("base_samples must be positive")
        if profile not in {"standard", "robust"}:
            raise ValueError("augmentation profile must be 'standard' or 'robust'")
        self.renderer = SyntheticRenderer(config, "train", global_seed)
        self.base_samples = base_samples
        self.global_seed = global_seed
        self.profile = profile
        self.transform = transform
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def __len__(self) -> int:
        return self.base_samples

    def raw_pair(self, index: int) -> tuple[Image.Image, Image.Image, int]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        upright = self.renderer.render(index)
        if self.profile == "robust":
            upright = apply_robust_augmentation(
                upright, augmentation_seed(self.global_seed, self.epoch, index)
            )
        upside_down = upright.transpose(Image.Transpose.ROTATE_180)
        target = index % 2
        return (upright, upside_down, target) if target == 0 else (upside_down, upright, target)

    def __getitem__(self, index: int) -> dict[str, object]:
        image, rotated, target = self.raw_pair(index)
        if self.transform is not None:
            image, rotated = self.transform(image), self.transform(rotated)
        return {"image": image, "rotated": rotated, "target": target}
