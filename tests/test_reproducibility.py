import random

import numpy as np
import torch

from src.reproducibility import make_generator, seed_everything, select_device


def draw_values() -> tuple[float, float, float]:
    return random.random(), float(np.random.random()), float(torch.rand(1).item())


def test_seed_everything_repeats_random_streams() -> None:
    seed_everything(123)
    first = draw_values()
    seed_everything(123)
    second = draw_values()

    assert first == second


def test_dataloader_generator_is_reproducible() -> None:
    first = torch.randperm(10, generator=make_generator(123))
    second = torch.randperm(10, generator=make_generator(123))

    assert torch.equal(first, second)


def test_device_selection_is_supported() -> None:
    assert select_device().type in {"cpu", "cuda"}
