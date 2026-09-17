from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from src.metrics import binary_metrics
from src.models import SmallOrientationCNN, count_parameters
from src.training import (
    evaluate_paired,
    load_checkpoint,
    paired_probabilities,
    save_checkpoint,
    train_epoch,
    train_paired_epoch,
)
from src.transforms import ResizePadToTensor


def test_resize_pad_preserves_shape_and_aspect_ratio() -> None:
    image = Image.new("RGB", (100, 20), "white")
    transform = ResizePadToTensor(height=96, width=384, fill=(0, 0, 0))

    tensor = transform(image)
    non_padding = (tensor > -0.99).any(dim=0).nonzero()
    content_height = int(non_padding[:, 0].max() - non_padding[:, 0].min() + 1)
    content_width = int(non_padding[:, 1].max() - non_padding[:, 1].min() + 1)

    assert tensor.shape == (3, 96, 384)
    assert abs(content_width / content_height - 5.0) < 0.1


def test_small_cnn_output_and_parameter_count() -> None:
    model = SmallOrientationCNN(dropout=0.0)
    logits = model(torch.zeros(2, 3, 96, 384))

    assert logits.shape == (2,)
    assert 100_000 < count_parameters(model) < 500_000


def test_binary_metrics() -> None:
    metrics = binary_metrics(np.array([0, 1]), np.array([0.1, 0.9]))

    assert np.isclose(metrics["brier_score"], 0.01)
    assert np.isclose(metrics["one_minus_brier"], 0.99)
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_training_updates_weights_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model = SmallOrientationCNN(dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    images = torch.rand(4, 3, 32, 64)
    targets = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(images, targets), batch_size=4)
    initial = model.classifier[-1].weight.detach().clone()

    loss = train_epoch(model, loader, optimizer, torch.device("cpu"), amp=False)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=1,
        metrics={"brier_score": 0.25},
        config={"seed": 7},
    )
    restored = SmallOrientationCNN(dropout=0.0)
    metadata = load_checkpoint(checkpoint_path, restored)

    assert np.isfinite(loss)
    assert not torch.equal(initial, model.classifier[-1].weight)
    assert metadata["epoch"] == 1
    model.eval()
    restored.eval()
    assert torch.equal(model(images), restored(images))


def test_paired_probability_is_complement_symmetric() -> None:
    direct = torch.tensor([-2.0, 1.0])
    rotated = torch.tensor([1.5, -0.5])
    _, _, prediction = paired_probabilities(direct, rotated)
    _, _, rotated_prediction = paired_probabilities(rotated, direct)

    assert torch.allclose(rotated_prediction, 1.0 - prediction)


def test_paired_train_and_evaluation() -> None:
    torch.manual_seed(11)
    model = SmallOrientationCNN(dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    direct = torch.rand(4, 3, 32, 64)
    rotated = torch.rot90(direct, 2, dims=(-2, -1))
    targets = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(
        [
            {"image": direct[i], "rotated": rotated[i], "target": targets[i]}
            for i in range(4)
        ],
        batch_size=4,
    )

    losses = train_paired_epoch(
        model,
        loader,
        optimizer,
        torch.device("cpu"),
        symmetry_loss_weight=0.1,
        amp=False,
    )
    metrics = evaluate_paired(model, loader, torch.device("cpu"))

    assert losses.keys() == {"loss", "classification_loss", "symmetry_loss"}
    assert 0 <= metrics["symmetric"]["brier_score"] <= 1
    assert metrics["mean_symmetry_error"] >= 0
