from pathlib import Path

import pytest
import torch

from src.recovery import inspect_recovery, load_recovery, recovery_fingerprint, save_recovery


def runtime(batch_size: int = 4) -> dict:
    return {
        "pretrained": False,
        "train_base_samples": 8,
        "validation_base_samples": 4,
        "num_workers": 0,
        "batch_size": batch_size,
        "validation_batch_size": 4,
        "frozen_epochs": 1,
        "finetune_epochs": 2,
    }


def test_recovery_fingerprint_changes_with_trajectory_setting() -> None:
    config = {"model": {"name": "small"}}
    assert recovery_fingerprint(config, runtime()) == recovery_fingerprint(config, runtime())
    assert recovery_fingerprint(config, runtime()) != recovery_fingerprint(config, runtime(8))


def test_recovery_round_trip_restores_training_state(tmp_path: Path) -> None:
    torch.manual_seed(9)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(17)
    state = {
        "phase": "finetune",
        "epoch": 3,
        "phase_epoch": 2,
        "best_brier": 0.2,
        "patience_used": 1,
        "history": [{"epoch": 3}],
    }
    fingerprint = recovery_fingerprint({"model": "small"}, runtime())
    expected_weight = model.weight.detach().clone()
    save_recovery(tmp_path, model, optimizer, state, fingerprint, generator)
    model.weight.data.zero_()

    restored = load_recovery(tmp_path, model, optimizer, fingerprint, generator)

    assert torch.equal(model.weight, expected_weight)
    assert restored["epoch"] == 3
    assert restored["history"] == [{"epoch": 3}]
    assert not restored["completed"]
    assert (tmp_path / "status.json").is_file()
    (tmp_path / "status.json").unlink()
    assert inspect_recovery(tmp_path, fingerprint)["phase"] == "finetune"


def test_recovery_rejects_incompatible_run(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    generator = torch.Generator().manual_seed(1)
    state = {
        "phase": "frozen",
        "epoch": 1,
        "phase_epoch": 1,
        "best_brier": 0.3,
        "patience_used": 0,
        "history": [],
    }
    save_recovery(tmp_path, model, optimizer, state, "first", generator)
    with pytest.raises(ValueError, match="incompatible"):
        load_recovery(tmp_path, model, optimizer, "second", generator)
