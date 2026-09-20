from pathlib import Path

import pytest
import torch

from scripts.train_robust import SUPPORTED_ROBUST_MODELS, _load_initial_weights


def save_checkpoint(path: Path, model: torch.nn.Module, model_name: str) -> None:
    torch.save(
        {
            "config": {"model": {"name": model_name}},
            "model_state": model.state_dict(),
            "epoch": 3,
        },
        path,
    )


def test_supported_robust_models_include_mobile_and_vit() -> None:
    assert SUPPORTED_ROBUST_MODELS == {"mobilenet_v3_large", "vit_b_16"}


def test_initial_checkpoint_must_match_selected_architecture(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    checkpoint = tmp_path / "mobile.pt"
    save_checkpoint(checkpoint, model, "mobilenet_v3_large")

    with pytest.raises(ValueError, match="does not match"):
        _load_initial_weights(model, checkpoint, "vit_b_16")


def test_matching_initial_checkpoint_is_loaded(tmp_path: Path) -> None:
    source = torch.nn.Linear(2, 1)
    target = torch.nn.Linear(2, 1)
    checkpoint = tmp_path / "vit.pt"
    save_checkpoint(checkpoint, source, "vit_b_16")

    metadata = _load_initial_weights(target, checkpoint, "vit_b_16")

    assert metadata["epoch"] == 3
    assert metadata["sha256"]
    for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
        assert torch.equal(source_parameter, target_parameter)
