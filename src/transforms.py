"""Image preprocessing with aspect-ratio-preserving resize and padding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class ResizePadToTensor:
    """Fit an RGB image into a fixed canvas without stretching or cropping."""

    height: int
    width: int
    fill: tuple[int, int, int] = (127, 127, 127)
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: tuple[float, float, float] = (0.5, 0.5, 0.5)

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError("height and width must be positive")
        if any(value <= 0 for value in self.std):
            raise ValueError("normalization std values must be positive")

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        scale = min(self.width / image.width, self.height / image.height)
        resized_width = max(1, min(self.width, round(image.width * scale)))
        resized_height = max(1, min(self.height, round(image.height * scale)))
        resized = image.resize(
            (resized_width, resized_height), Image.Resampling.BILINEAR
        )
        canvas = Image.new("RGB", (self.width, self.height), self.fill)
        x = (self.width - resized_width) // 2
        y = (self.height - resized_height) // 2
        canvas.paste(resized, (x, y))

        array = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
        tensor = torch.from_numpy(array)
        mean = torch.tensor(self.mean, dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor(self.std, dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std
