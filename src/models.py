"""Model definitions and construction."""

from __future__ import annotations

import math
from collections import OrderedDict
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    ViT_B_16_Weights,
    VisionTransformer,
    efficientnet_b0,
    mobilenet_v3_large,
)


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


class BinaryLinear(nn.Linear):
    """Linear binary head with a stable `[batch]` logit contract."""

    def __init__(self, in_features: int):
        super().__init__(in_features, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return super().forward(features).squeeze(-1)


class RectangularVisionTransformer(VisionTransformer):
    """Torchvision ViT-B/16 with a fixed rectangular patch grid."""

    def __init__(self, image_height: int, image_width: int, dropout: float = 0.0):
        patch_size = 16
        if image_height % patch_size or image_width % patch_size:
            raise ValueError("ViT input dimensions must be divisible by patch size 16")
        super().__init__(
            image_size=image_height,
            patch_size=patch_size,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            dropout=dropout,
            attention_dropout=0.0,
            num_classes=1000,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        self.image_height = image_height
        self.image_width = image_width
        token_count = (image_height // patch_size) * (image_width // patch_size) + 1
        self.encoder.pos_embedding = nn.Parameter(
            torch.empty(1, token_count, self.hidden_dim).normal_(std=0.02)
        )

    def _process_input(self, images: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = images.shape
        torch._assert(
            height == self.image_height,
            f"Wrong image height! Expected {self.image_height} but got {height}!",
        )
        torch._assert(
            width == self.image_width,
            f"Wrong image width! Expected {self.image_width} but got {width}!",
        )
        patches = self.conv_proj(images)
        patches = patches.reshape(batch, self.hidden_dim, -1)
        return patches.permute(0, 2, 1)


def interpolate_vit_positional_embedding(
    state_dict: dict[str, torch.Tensor],
    target_height: int,
    target_width: int,
    patch_size: int = 16,
) -> OrderedDict[str, torch.Tensor]:
    """Resize square pretrained patch positions while preserving the CLS token."""
    if target_height % patch_size or target_width % patch_size:
        raise ValueError("ViT target dimensions must be divisible by patch size")
    result = OrderedDict(state_dict)
    positional = result["encoder.pos_embedding"]
    if positional.ndim != 3 or positional.shape[0] != 1:
        raise ValueError(f"Unexpected position embedding shape: {positional.shape}")
    cls_position, patch_positions = positional[:, :1], positional[:, 1:]
    source_side = math.isqrt(patch_positions.shape[1])
    if source_side * source_side != patch_positions.shape[1]:
        raise ValueError("pretrained ViT patch positions do not form a square grid")
    target_grid = (target_height // patch_size, target_width // patch_size)
    patch_positions = patch_positions.permute(0, 2, 1).reshape(
        1, positional.shape[2], source_side, source_side
    )
    patch_positions = F.interpolate(
        patch_positions,
        size=target_grid,
        mode="bicubic",
        align_corners=True,
    )
    patch_positions = patch_positions.flatten(2).permute(0, 2, 1)
    result["encoder.pos_embedding"] = torch.cat(
        (cls_position, patch_positions), dim=1
    )
    return result


def _build_vit_b_16(
    dropout: float,
    pretrained: bool,
    input_height: int,
    input_width: int,
) -> RectangularVisionTransformer:
    model = RectangularVisionTransformer(input_height, input_width, dropout)
    if pretrained:
        weights = ViT_B_16_Weights.DEFAULT
        state_dict = weights.get_state_dict(progress=True, check_hash=True)
        state_dict = interpolate_vit_positional_embedding(
            state_dict, input_height, input_width
        )
        model.load_state_dict(state_dict)
    model.heads.head = BinaryLinear(model.heads.head.in_features)
    return model


def build_model(
    name: str,
    dropout: float = 0.2,
    pretrained: bool = False,
    input_height: int = 224,
    input_width: int = 224,
) -> nn.Module:
    """Construct a model available at the current implementation stage."""
    if name == "small_cnn":
        return SmallOrientationCNN(dropout=dropout)
    if name == "mobilenet_v3_large":
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_large(weights=weights, dropout=dropout)
        model.classifier[-1] = BinaryLinear(model.classifier[-1].in_features)
        return model
    if name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights, dropout=dropout)
        model.classifier[-1] = BinaryLinear(model.classifier[-1].in_features)
        return model
    if name == "vit_b_16":
        return _build_vit_b_16(
            dropout, pretrained, input_height, input_width
        )
    raise ValueError(f"Model '{name}' is configured but not implemented at this stage")


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    parameters = (parameter for parameter in model.parameters() if parameter.requires_grad)
    if not trainable_only:
        parameters = model.parameters()
    return sum(parameter.numel() for parameter in parameters)


def freeze_backbone(model: nn.Module) -> None:
    """Freeze a CNN/ViT backbone while keeping only its binary head trainable."""
    features = getattr(model, "features", None)
    classifier = getattr(model, "classifier", None)
    heads = getattr(model, "heads", None)
    head = classifier if classifier is not None else heads
    if head is None or (features is None and heads is None):
        raise ValueError("Model does not expose a supported classification head")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in head.parameters():
        parameter.requires_grad = True


def unfreeze_model(model: nn.Module) -> None:
    """Enable gradients for every model parameter."""
    for parameter in model.parameters():
        parameter.requires_grad = True
