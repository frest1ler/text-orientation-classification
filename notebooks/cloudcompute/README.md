# CloudCompute / Linux GPU notebooks

These notebooks run without Google Colab or Google Drive. They assume a Linux
GPU machine with Jupyter and preserve data separately from the Git checkout:

```python
REPO_DIR = "/root/text-orientation-classification"
STATE_DIR = "/root/text-orientation-state"
```

Clone the repository into `REPO_DIR`. Put persistent files in `STATE_DIR`:

```text
/root/text-orientation-state/
├── data/test.zip
├── registry/
├── training/recovery/
├── training/runs/
├── tuning/optuna/
├── evaluation/
└── inference/runs/
```

Deleting the rented instance may delete `STATE_DIR`; copy the registry,
recovery state, run archives, and submissions elsewhere before termination.

## Standard training: `train.ipynb`

For a full ViT run or resume:

```python
MODEL = "vit"                 # mobilenet | efficientnet | vit
QUICK_RUN = False
PROMOTE_CHAMPION = True
RUN_CALIBRATION = True
RESUME_TRAINING = True
RUN_TESTS = False
TRAIN_BATCH_SIZE = 16         # use 8 after CUDA OOM
VALIDATION_BATCH_SIZE = 32    # use 16 after CUDA OOM
```

For MobileNet/EfficientNet, `None` uses the config defaults. To resume, the
matching directory must already contain `last.pt`, for example:
`/root/text-orientation-state/training/recovery/vit_b_16/full/last.pt`.
Do not change model, quick/full mode, or configuration when resuming.

## Robust training: `robust_train.ipynb`

The standard champion must already be present in `STATE_DIR/registry`.

```python
MODEL = "vit"                    # vit | mobilenet
QUICK_RUN = False
AUGMENTATION_PROFILE = "robust"
EPOCHS = 3
TRAIN_BATCH_SIZE = 16            # MobileNet: 64
VALIDATION_BATCH_SIZE = 32       # MobileNet: 128
RESUME_TRAINING = True
PROMOTE_ROBUST_CHAMPION = True
RUN_TESTS = False
```

This fine-tunes from the selected standard champion and writes to an isolated
robust recovery/registry namespace.

## Inference: `inference.ipynb`

Place `test.zip` at `/root/text-orientation-state/data/test.zip` and copy the
required registry into `/root/text-orientation-state/registry`. Configure:

```python
MODEL = "best_ensemble"  # or best / named standard / named robust model
BATCH_SIZE = None
NUM_WORKERS = 2
RESUME_INFERENCE = True
SMOKE_IMAGES = None      # 64 only for a smoke check
RUN_TESTS = False
```

The result is printed and stored below
`/root/text-orientation-state/inference/runs/<run>/submission.csv`.

## Ensemble: `ensemble.ipynb`

Edit `CANDIDATES` so every path points to a completed calibrated run containing
`best.pt`, `calibration.json`, and `validation_predictions.npz`. Use
`WEIGHT_STEP=0.05` and `PROMOTE_ENSEMBLE=True` for the final search. No model is
retrained; only validation probabilities are combined.

## Optuna: `optuna.ipynb`

Use `SMOKE_RUN=True` first. For the bounded experiment use
`SMOKE_RUN=False`, `TRIALS=8` (up to 12 if time permits), and keep the supplied
sample budgets. Results and the SQLite study are saved under
`STATE_DIR/tuning/optuna`, so the search can resume after restarting Jupyter.

## Moving results off the server

Before stopping the instance, copy at least:

- `STATE_DIR/registry/` — selectable champions and ensemble manifests;
- `STATE_DIR/training/recovery/` — resumable epoch checkpoints;
- `STATE_DIR/training/runs/` — calibrated run outputs;
- `STATE_DIR/inference/runs/*/submission.csv` — predictions;
- `STATE_DIR/tuning/optuna/` — Optuna database and reports.

Model checkpoints are intentionally not stored in Git. Use `rsync`, `scp`, or
an authenticated cloud client to transfer them.
