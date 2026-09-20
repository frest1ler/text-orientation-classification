"""Run a resumable, registry-isolated Optuna search for MobileNetV3-Large."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import torch

from scripts.train import make_loader
from src.config import Config, load_config
from src.models import build_model, freeze_backbone, unfreeze_model
from src.reproducibility import seed_everything, select_device
from src.synthetic import PairedSyntheticDataset
from src.training import evaluate_paired, save_checkpoint, train_paired_epoch
from src.transforms import build_preprocess
from src.tuning import (
    TuningBudget,
    apply_trial_parameters,
    confirmation_config,
    export_trials_csv,
    lock_search_protocol,
    promote_trial_checkpoint,
    suggest_parameters,
    tuning_root,
    write_json_atomic,
    write_yaml_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--study-name", default="mobilenet_v3_large_deadline_search")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--train-base-samples", type=int, default=8000)
    parser.add_argument("--validation-base-samples", type=int, default=5000)
    parser.add_argument("--frozen-epochs", type=int, default=1)
    parser.add_argument("--finetune-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--validation-batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _run_trial(
    trial: Any,
    base_config: Config,
    budget: TuningBudget,
    output_root: Path,
    batch_size: int,
    validation_batch_size: int,
    num_workers: int,
    optuna: Any,
) -> float:
    parameters = suggest_parameters(trial)
    config = apply_trial_parameters(base_config, parameters)
    # Every trial sees the same samples, order, initialisation seed, and validation set.
    seed_everything(config.experiment.seed, config.training.deterministic)
    device = select_device()
    transform = build_preprocess(
        config.model.name, config.model.input_height, config.model.input_width
    )
    train_dataset = PairedSyntheticDataset(
        config.synthetic, "train", budget.train_samples, config.experiment.seed, transform
    )
    validation_dataset = PairedSyntheticDataset(
        config.synthetic,
        "validation",
        budget.validation_samples,
        config.experiment.seed,
        transform,
    )
    train_loader = make_loader(
        train_dataset, batch_size, True, num_workers, config.experiment.seed, device.type == "cuda"
    )
    validation_loader = make_loader(
        validation_dataset,
        validation_batch_size,
        False,
        num_workers,
        config.experiment.seed,
        device.type == "cuda",
    )
    model = build_model(
        config.model.name,
        dropout=config.model.dropout,
        pretrained=config.model.pretrained,
        input_height=config.model.input_height,
        input_width=config.model.input_width,
    ).to(device)
    history: list[dict[str, Any]] = []
    best_brier = float("inf")
    best_metrics: dict[str, Any] | None = None
    epoch_number = 0
    trial_dir = output_root / "runs" / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(trial_dir / "params.json", parameters)

    with tempfile.TemporaryDirectory(prefix=f"optuna-trial-{trial.number}-") as temporary:
        checkpoint = Path(temporary) / "best.pt"
        phases = (("frozen", budget.frozen_epochs), ("finetune", budget.finetune_epochs))
        for phase, epochs in phases:
            if epochs == 0:
                continue
            if phase == "frozen":
                freeze_backbone(model)
                optimizer = torch.optim.AdamW(
                    (p for p in model.parameters() if p.requires_grad),
                    lr=config.training.frozen_learning_rate,
                    weight_decay=config.training.weight_decay,
                )
            else:
                unfreeze_model(model)
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=config.training.finetune_learning_rate,
                    weight_decay=config.training.weight_decay,
                )
            for phase_epoch in range(1, epochs + 1):
                epoch_number += 1
                losses = train_paired_epoch(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    config.training.symmetry_loss_weight,
                    config.training.amp,
                    f"trial {trial.number} {phase} {phase_epoch}/{epochs} train",
                )
                metrics = evaluate_paired(
                    model,
                    validation_loader,
                    device,
                    f"trial {trial.number} {phase} {phase_epoch}/{epochs} validation",
                )
                brier = float(metrics["symmetric"]["brier_score"])
                history.append(
                    {"epoch": epoch_number, "phase": phase, **losses, **metrics}
                )
                write_json_atomic(trial_dir / "history.json", history)
                if brier < best_brier:
                    best_brier, best_metrics = brier, metrics
                    save_checkpoint(
                        checkpoint, model, optimizer, epoch_number, metrics, config.to_dict()
                    )
                trial.report(brier, epoch_number)
                if trial.should_prune():
                    trial.set_user_attr("best_brier", best_brier)
                    raise optuna.TrialPruned(f"pruned at epoch {epoch_number}")

        if best_metrics is None:
            raise RuntimeError("trial completed without validation metrics")
        # Completed trials race only against earlier completed trials. The search is
        # intentionally sequential, so the tuning-only checkpoint remains unambiguous.
        previous = min(
            (
                completed.value
                for completed in trial.study.get_trials(deepcopy=False)
                if completed.state.name == "COMPLETE"
                and completed.value is not None
                and completed.number != trial.number
            ),
            default=float("inf"),
        )
        if best_brier < previous:
            promote_trial_checkpoint(
                checkpoint,
                output_root / "best_trial",
                trial.number,
                best_brier,
                parameters,
            )
        trial.set_user_attr("best_brier", best_brier)
        trial.set_user_attr("best_epoch", int(torch.load(checkpoint, map_location="cpu", weights_only=True)["epoch"]))
        trial.set_user_attr("symmetric_accuracy", float(best_metrics["symmetric"]["accuracy"]))
        trial.set_user_attr("symmetric_roc_auc", float(best_metrics["symmetric"]["roc_auc"]))
        return best_brier


def _trial_rows(study: Any) -> list[dict[str, Any]]:
    rows = []
    for trial in study.trials:
        row: dict[str, Any] = {
            "number": trial.number,
            "state": trial.state.name,
            "symmetric_brier_score": trial.value,
        }
        row.update({f"param_{key}": value for key, value in trial.params.items()})
        row.update({f"metric_{key}": value for key, value in trial.user_attrs.items()})
        rows.append(row)
    return rows


def main() -> None:
    try:
        import optuna
    except ImportError as error:
        raise RuntimeError("Install project requirements to use Optuna") from error

    args = parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    base_config = load_config(args.config)
    if base_config.model.name != "mobilenet_v3_large":
        raise ValueError("The deadline search is intentionally limited to MobileNetV3-Large")
    budget = TuningBudget(
        train_samples=1024 if args.smoke else args.train_base_samples,
        validation_samples=512 if args.smoke else args.validation_base_samples,
        frozen_epochs=1 if args.smoke else args.frozen_epochs,
        finetune_epochs=1 if args.smoke else args.finetune_epochs,
    )
    budget.validate()
    trials = min(args.trials, 2) if args.smoke else args.trials
    root = tuning_root(args.project_dir, base_config.model.name) / (
        "smoke" if args.smoke else "search"
    )
    root.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(root / 'study.db').resolve()}"
    sampler = optuna.samplers.TPESampler(seed=base_config.experiment.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )
    batch_size = args.batch_size or base_config.training.batch_size
    validation_batch_size = args.validation_batch_size or base_config.inference.batch_size
    num_workers = base_config.data.num_workers if args.num_workers is None else args.num_workers
    protocol = {
        "study_name": args.study_name,
        "model": base_config.model.name,
        "seed": base_config.experiment.seed,
        "budget": vars(budget),
        "batch_size": batch_size,
        "validation_batch_size": validation_batch_size,
        "num_workers": num_workers,
        "search_space_version": 1,
    }
    lock_search_protocol(root / "search_protocol.json", protocol)
    objective = lambda trial: _run_trial(
        trial,
        base_config,
        budget,
        root,
        batch_size,
        validation_batch_size,
        num_workers,
        optuna,
    )
    terminal_states = {"COMPLETE", "PRUNED", "FAIL"}
    attempted = sum(trial.state.name in terminal_states for trial in study.trials)
    remaining = max(0, trials - attempted)
    if remaining:
        study.optimize(
            objective, n_trials=remaining, gc_after_trial=True, show_progress_bar=True
        )
    else:
        print(f"Study already has {attempted} terminal trials; no new trials requested.")
    export_trials_csv(root / "trials.csv", _trial_rows(study))
    best_parameters = {key: float(value) for key, value in study.best_params.items()}
    write_json_atomic(root / "best_params.json", best_parameters)
    write_yaml_atomic(
        root / "best_config.yaml", confirmation_config(base_config, best_parameters)
    )
    summary = {
        **protocol,
        "target_trials_total": trials,
        "attempted_trials_total": sum(
            trial.state.name in terminal_states for trial in study.trials
        ),
        "completed_trials_total": sum(t.state.name == "COMPLETE" for t in study.trials),
        "best_trial": study.best_trial.number,
        "best_symmetric_brier_score": study.best_value,
        "best_parameters": best_parameters,
        "confirmation_config": str(root / "best_config.yaml"),
    }
    write_json_atomic(root / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
