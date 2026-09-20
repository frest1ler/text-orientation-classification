# Deadline-friendly Optuna search

The search is deliberately restricted to MobileNetV3-Large and four
parameters: fine-tuning learning rate, weight decay, classifier dropout, and
symmetry-loss weight. Test data, model architecture, image size, batch size,
and validation samples are not tuned.

All persistent artifacts are isolated from the champion registry:

```text
text-orientation/tuning/optuna/mobilenet_v3_large/
  smoke/
  search/
    study.db
    search_config.json
    trials.csv
    best_params.json
    best_config.yaml
    summary.json
    best_trial/
      best.pt
      best_trial.json
    runs/trial_XXXX/
      params.json
      history.json
```

Only one tuning checkpoint is retained. `study.db` resumes completed trials
after a Colab interruption, and `N_TRIALS` is the target total rather than the
number added by every rerun. An immutable protocol rejects changed budgets.
Smoke and main studies are separated so they can never be ranked together.

## Colab flow

Open `colab_optuna.ipynb` and first use `SMOKE_RUN=True`. This runs at most two
trials with 1,024 train and 512 validation samples and two total epochs. Then
set `SMOKE_RUN=False`; the recommended deadline budget is 8–12 trials, 8,000
train samples, the locked 5,000-sample validation set, one frozen epoch, and
three fine-tuning epochs.

The objective is minimum symmetric validation Brier score. A seeded TPE
sampler, median pruning, identical synthetic samples, and identical model and
loader seeds keep comparisons controlled. Trials never read `test.zip`.

## Full confirmation

`best_config.yaml` contains the original full-data schedule with only the four
winning values changed. It is a candidate, not a champion. Confirm it with:

```bash
python3 -m scripts.train \
  --config /path/to/tuning/optuna/mobilenet_v3_large/search/best_config.yaml \
  --run-name mobilenet_optuna_confirmation \
  --recovery-dir /path/to/text-orientation/training/recovery/mobilenet_optuna/full \
  --resume
```

After completion, run calibration and then `scripts.promote_champion`. Existing
promotion safeguards retain the current MobileNet champion when the confirmed
candidate does not improve comparable symmetric Brier. An unconfirmed short
trial must never be promoted.

The notebook automates the same flow when `SMOKE_RUN=False` and
`RUN_FULL_CONFIRMATION=True`: resumable full training, calibration, guarded
promotion, and archival under `training/runs/mobilenet_v3_large/`. Keep the
flag false until the main search report has been reviewed.
