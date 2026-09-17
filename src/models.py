"""Model definitions and construction."""

from __future__ import annotations

import torch
from torch import nn


class ConvNormActivation(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU(inplace=True),
        )


class SmallOrientationCNN(nn.Module):
    """Low-cost pipeline baseline, not intended as the final model."""

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            ConvNormActivation(3, 16, stride=2),
            ConvNormActivation(16, 32, stride=2),
            ConvNormActivation(32, 64, stride=2),
            ConvNormActivation(64, 96, stride=2),
            ConvNormActivation(96, 128, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.classifier(self.pool(features)).squeeze(1)


def build_model(name: str, dropout: float = 0.2) -> nn.Module:
    """Construct a model available at the current implementation stage."""
    if name == "small_cnn":
        return SmallOrientationCNN(dropout=dropout)
    raise ValueError(f"Model '{name}' is configured but not implemented yet")


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    parameters = (parameter for parameter in model.parameters() if parameter.requires_grad)
    if not trainable_only:
        parameters = model.parameters()
    return sum(parameter.numel() for parameter in parameters)
