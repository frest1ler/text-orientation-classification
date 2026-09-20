from pathlib import Path

import numpy as np
from PIL import Image

from src.config import load_config
from src.robust_augmentation import (
    ProfiledPairedSyntheticDataset,
    apply_robust_augmentation,
    augmentation_seed,
)
from src.synthetic import PairedSyntheticDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_robust_augmentation_is_seeded_rgb_and_shape_preserving() -> None:
    image = Image.new("RGB", (160, 48), (120, 130, 140))
    first = apply_robust_augmentation(image, 123)
    second = apply_robust_augmentation(image, 123)

    assert first.mode == "RGB"
    assert first.size == image.size
    assert np.array_equal(np.asarray(first), np.asarray(second))


def test_augmentation_seed_changes_by_epoch_and_sample() -> None:
    assert augmentation_seed(42, 1, 3) == augmentation_seed(42, 1, 3)
    assert augmentation_seed(42, 1, 3) != augmentation_seed(42, 2, 3)
    assert augmentation_seed(42, 1, 3) != augmentation_seed(42, 1, 4)


def test_standard_profile_matches_existing_train_dataset() -> None:
    config = load_config(PROJECT_ROOT / "configs/baseline.yaml")
    profiled = ProfiledPairedSyntheticDataset(
        config.synthetic, 2, config.experiment.seed, "standard"
    )
    existing = PairedSyntheticDataset(
        config.synthetic, "train", 2, config.experiment.seed
    )

    for index in range(2):
        image, rotated, target = profiled.raw_pair(index)
        original = existing[index]
        assert target == original["target"]
        assert np.array_equal(np.asarray(image), np.asarray(original["image"]))
        assert np.array_equal(np.asarray(rotated), np.asarray(original["rotated"]))


def test_robust_pair_is_exact_rotation_and_changes_across_epochs() -> None:
    config = load_config(PROJECT_ROOT / "configs/baseline.yaml")
    dataset = ProfiledPairedSyntheticDataset(
        config.synthetic, 12, config.experiment.seed, "robust"
    )
    epoch_zero = []
    epoch_one = []
    for epoch, collection in ((0, epoch_zero), (1, epoch_one)):
        dataset.set_epoch(epoch)
        for index in range(len(dataset)):
            image, rotated, target = dataset.raw_pair(index)
            expected = image.transpose(Image.Transpose.ROTATE_180)
            assert np.array_equal(np.asarray(rotated), np.asarray(expected))
            assert target == index % 2
            collection.append(np.asarray(image))

    assert any(
        not np.array_equal(first, second)
        for first, second in zip(epoch_zero, epoch_one, strict=True)
    )


def test_validation_remains_epoch_independent() -> None:
    config = load_config(PROJECT_ROOT / "configs/baseline.yaml")
    validation = PairedSyntheticDataset(
        config.synthetic, "validation", 1, config.experiment.seed
    )
    first = validation[0]
    second = validation[0]
    assert np.array_equal(np.asarray(first["image"]), np.asarray(second["image"]))

