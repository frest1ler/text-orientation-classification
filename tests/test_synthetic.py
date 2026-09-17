import hashlib

import numpy as np
from PIL import Image

from src.config import load_config
from src.synthetic import (
    SyntheticOrientationDataset,
    SyntheticRenderer,
    sample_target_size,
    sample_seed,
    verify_font_assets,
)
from src.text_corpus import TRAIN_WORDS, VALIDATION_WORDS


def image_hash(image: Image.Image) -> str:
    payload = image.mode.encode() + str(image.size).encode() + np.asarray(image).tobytes()
    return hashlib.sha256(payload).hexdigest()


def test_bundled_font_checksums_and_splits() -> None:
    config = load_config("configs/baseline.yaml")
    fonts = verify_font_assets(config.synthetic.font_assets_dir)

    assert len(fonts["train"]) == 8
    assert len(fonts["validation"]) == 4
    assert set(fonts["train"]).isdisjoint(fonts["validation"])
    assert {path.stem.split("-")[0] for path in fonts["train"]} == {
        "DejaVuSans",
        "DejaVuSerif",
        "NotoSans",
        "NotoSerif",
    }
    assert {path.stem.split("-")[0] for path in fonts["validation"]} == {
        "LiberationSans",
        "LiberationSerif",
    }


def test_word_corpora_do_not_leak_between_splits() -> None:
    for language in ("ru", "en"):
        assert set(TRAIN_WORDS[language]).isdisjoint(VALIDATION_WORDS[language])


def test_sample_seed_depends_on_split_and_sample() -> None:
    assert sample_seed(42, "train", 0) == sample_seed(42, "train", 0)
    assert sample_seed(42, "train", 0) != sample_seed(42, "train", 1)
    assert sample_seed(42, "train", 0) != sample_seed(42, "validation", 0)


def test_renderer_is_order_independent() -> None:
    config = load_config("configs/baseline.yaml")
    first_renderer = SyntheticRenderer(config.synthetic, "train", global_seed=42)
    first = image_hash(first_renderer.render(7))
    first_renderer.render(3)
    repeated = image_hash(first_renderer.render(7))
    second_renderer = SyntheticRenderer(config.synthetic, "train", global_seed=42)

    assert first == repeated == image_hash(second_renderer.render(7))


def test_dataset_pairs_are_exact_rotations_and_balanced() -> None:
    config = load_config("configs/baseline.yaml")
    dataset = SyntheticOrientationDataset(
        config.synthetic, "validation", base_samples=3, global_seed=42
    )

    labels = []
    for pair_index in range(3):
        upright, upright_label = dataset[pair_index * 2]
        rotated, rotated_label = dataset[pair_index * 2 + 1]
        labels.extend((upright_label, rotated_label))
        expected = upright.transpose(Image.Transpose.ROTATE_180)
        assert np.array_equal(np.asarray(rotated), np.asarray(expected))

    assert labels.count(0) == labels.count(1) == 3
    assert len(dataset) == 6


def test_target_geometry_matches_audit_distribution() -> None:
    config = load_config("configs/baseline.yaml")
    rng = np.random.default_rng(42)
    sizes = np.asarray([sample_target_size(rng, config.synthetic) for _ in range(5_000)])
    aspect_ratios = sizes[:, 0] / sizes[:, 1]

    assert 4.5 < float(np.median(aspect_ratios)) < 5.3
    assert 0.19 < float((aspect_ratios > 8).mean()) < 0.26
    assert sizes[:, 0].min() >= 25
    assert sizes[:, 0].max() <= 1599


def test_rendering_snapshots() -> None:
    config = load_config("configs/baseline.yaml")
    expected = {
        ("train", 0): "8fe7581f9c2e201561f6b85cd2a19ab293d6ad1edaf48310cb6088c067334ef5",
        ("train", 1): "f022188f4c42e5b2f55c3340aa2ea4f8625d8829be688c60b0bbc283c7e136cc",
        ("train", 7): "9a864708b13978b6aee5334040421aa2b4f91c149d79ef43a361d60d27e20fba",
        ("validation", 0): "1e9149b443c7a6ddbe31c65e28a7dc842b963fc11834be248382e32701f1413a",
        ("validation", 1): "9ee3130b7aade9153eb83341ff39dae10aa2f9288fc279fa66057c5c8e02b661",
        ("validation", 7): "00725fe4380383ad43dd8c414cffd356b763ab73f05b947560c99834a945c4e2",
    }
    for split in ("train", "validation"):
        renderer = SyntheticRenderer(config.synthetic, split, global_seed=42)
        for sample_id in (0, 1, 7):
            assert image_hash(renderer.render(sample_id)) == expected[(split, sample_id)]
