import csv
import json
from pathlib import Path

import pytest
import yaml

from src.config import load_config
from src.tuning import (
    TuningBudget,
    apply_trial_parameters,
    confirmation_config,
    export_trials_csv,
    lock_search_protocol,
    promote_trial_checkpoint,
    suggest_parameters,
    tuning_root,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = {
    "finetune_learning_rate": 5e-5,
    "weight_decay": 2e-4,
    "dropout": 0.3,
    "symmetry_loss_weight": 0.2,
}


class FakeTrial:
    def suggest_float(self, name, low, high, log=False):
        del log
        return (low + high) / 2


def test_tuning_root_is_outside_registry(tmp_path: Path) -> None:
    root = tuning_root(tmp_path, "mobilenet_v3_large")

    assert root == tmp_path / "tuning/optuna/mobilenet_v3_large"
    assert tmp_path / "registry" not in root.parents


def test_apply_trial_parameters_changes_only_search_fields() -> None:
    base = load_config(PROJECT_ROOT / "configs/baseline.yaml")
    tuned = apply_trial_parameters(base, PARAMETERS)

    assert tuned.model.dropout == 0.3
    assert tuned.training.finetune_learning_rate == 5e-5
    assert tuned.training.weight_decay == 2e-4
    assert tuned.training.symmetry_loss_weight == 0.2
    assert tuned.experiment == base.experiment
    assert tuned.synthetic == base.synthetic
    assert tuned.training.frozen_learning_rate == base.training.frozen_learning_rate


def test_apply_trial_parameters_rejects_uncontrolled_space() -> None:
    base = load_config(PROJECT_ROOT / "configs/baseline.yaml")

    with pytest.raises(ValueError, match="exactly"):
        apply_trial_parameters(base, {**PARAMETERS, "batch_size": 32})


def test_suggest_parameters_uses_only_four_fields() -> None:
    assert set(suggest_parameters(FakeTrial())) == set(PARAMETERS)


def test_budget_validation() -> None:
    TuningBudget(8000, 5000).validate()
    with pytest.raises(ValueError, match="sample counts"):
        TuningBudget(0, 5000).validate()


def test_best_checkpoint_replaces_one_tuning_only_file(tmp_path: Path) -> None:
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    destination = tmp_path / "tuning-best"

    promote_trial_checkpoint(first, destination, 1, 0.2, PARAMETERS)
    promote_trial_checkpoint(second, destination, 2, 0.1, PARAMETERS)

    assert (destination / "best.pt").read_bytes() == b"second"
    metadata = json.loads((destination / "best_trial.json").read_text())
    assert metadata["trial"] == 2
    assert list(destination.glob("*.pt")) == [destination / "best.pt"]


def test_exports_are_confirmation_ready(tmp_path: Path) -> None:
    base = load_config(PROJECT_ROOT / "configs/baseline.yaml")
    config = confirmation_config(base, PARAMETERS)
    config_path = tmp_path / "best_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    loaded = load_config(config_path)
    assert loaded.model.dropout == PARAMETERS["dropout"]

    csv_path = export_trials_csv(
        tmp_path / "trials.csv",
        [{"number": 0, "state": "COMPLETE", "score": 0.2}],
    )
    with csv_path.open(newline="") as stream:
        assert list(csv.DictReader(stream))[0]["state"] == "COMPLETE"


def test_search_protocol_prevents_mixed_budgets(tmp_path: Path) -> None:
    path = tmp_path / "search_protocol.json"
    protocol = {"budget": {"train_samples": 8000}, "seed": 42}
    lock_search_protocol(path, protocol)
    lock_search_protocol(path, protocol)

    with pytest.raises(ValueError, match="protocol differs"):
        lock_search_protocol(path, {"budget": {"train_samples": 4000}, "seed": 42})
