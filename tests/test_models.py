import pytest
import torch

from src.models import (
    build_model,
    count_parameters,
    freeze_backbone,
    interpolate_vit_positional_embedding,
    unfreeze_model,
)
from src.transforms import IMAGENET_MEAN, IMAGENET_STD, build_preprocess


@pytest.mark.parametrize(
    ("name", "minimum_parameters", "maximum_parameters"),
    [
        ("mobilenet_v3_large", 4_000_000, 6_000_000),
        ("efficientnet_b0", 3_500_000, 6_000_000),
    ],
)
def test_compact_model_forward_without_pretrained_download(
    name: str, minimum_parameters: int, maximum_parameters: int
) -> None:
    model = build_model(name, dropout=0.2, pretrained=False)
    model.eval()
    with torch.inference_mode():
        logits = model(torch.zeros(2, 3, 64, 128))

    assert logits.shape == (2,)
    assert minimum_parameters < count_parameters(model) < maximum_parameters


@pytest.mark.parametrize("name", ["mobilenet_v3_large", "efficientnet_b0"])
def test_freeze_and_unfreeze_backbone(name: str) -> None:
    model = build_model(name, pretrained=False)

    freeze_backbone(model)

    assert not any(parameter.requires_grad for parameter in model.features.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    assert 0 < count_parameters(model, trainable_only=True) < count_parameters(model)

    unfreeze_model(model)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_pretrained_models_use_imagenet_normalization() -> None:
    transform = build_preprocess("mobilenet_v3_large", height=96, width=384)

    assert transform.mean == IMAGENET_MEAN
    assert transform.std == IMAGENET_STD


def test_rectangular_vit_forward_and_freezing() -> None:
    model = build_model(
        "vit_b_16",
        dropout=0.1,
        pretrained=False,
        input_height=96,
        input_width=384,
    )
    model.eval()
    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, 96, 384))

    assert logits.shape == (1,)
    assert model.encoder.pos_embedding.shape == (1, 145, 768)
    assert 85_000_000 < count_parameters(model) < 87_000_000
    freeze_backbone(model)
    assert all(parameter.requires_grad for parameter in model.heads.parameters())
    assert count_parameters(model, trainable_only=True) == 769
    unfreeze_model(model)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_vit_position_interpolation_preserves_cls_and_targets_24_by_6_grid() -> None:
    positional = torch.arange(197 * 4, dtype=torch.float32).reshape(1, 197, 4)
    result = interpolate_vit_positional_embedding(
        {"encoder.pos_embedding": positional},
        target_height=96,
        target_width=384,
    )

    assert result["encoder.pos_embedding"].shape == (1, 145, 4)
    assert torch.equal(result["encoder.pos_embedding"][:, :1], positional[:, :1])
