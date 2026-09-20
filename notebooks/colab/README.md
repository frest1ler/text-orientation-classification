# Google Colab notebooks

This directory contains thin launchers around the tested code in `src/` and
`scripts/`. Use a GPU runtime and run cells from top to bottom. Do not run two
training notebooks against the same recovery directory at the same time.

## 1. Reproduce the submitted solution

Open `solution.ipynb`. This is the reviewer-facing notebook and does **not**
require a pre-existing Google Drive project.

Keep these pinned values unchanged:

```python
REVISION = "solution-v1"
ARTIFACT_URL = "https://github.com/frest1ler/text-orientation-classification/releases/download/solution-v1/text-orientation-solution-artifacts.zip"
ARTIFACT_SHA256 = "08864ad16e7df88489c6479a060c817d9d9041a6c75674f077a7167954583502"
TEST_ZIP_PATH = "/content/test.zip"
BATCH_SIZE = None
```

Procedure:

1. Select **Runtime → Change runtime type → GPU**.
2. Upload the issued archive as `/content/test.zip`. If it is absent, the
   notebook opens an upload dialog automatically.
3. Select **Runtime → Run all**.
4. Wait for both ensemble components. They run sequentially to reduce GPU RAM.
5. The notebook validates 20,000 rows and downloads `submission.csv`.

The final predictor is `0.25 × robust MobileNetV3-Large + 0.75 × robust
ViT-B/16`. For the original archive the expected submission SHA-256 is
`cf3be491f4df56015bce7f78f5db9c35ea7ad5f651d6fdf801ebff99fd1a5482`.

## 2. Persistent Google Drive layout

All experiment notebooks use:

```python
PROJECT_DIR = "/content/drive/MyDrive/text-orientation"
```

The minimum useful layout is:

```text
text-orientation/
├── data/test.zip
├── registry/
│   ├── leaderboard.json
│   ├── champions/<model>/...
│   ├── robust/...
│   └── ensembles/...
├── training/
│   ├── recovery/...
│   └── runs/...
├── tuning/optuna/...
├── evaluation/...
└── inference/runs/...
```

`data/test.zip` contains the provided unlabeled images. Training labels are
generated deterministically from synthetic upright/180-degree pairs; the test
archive is never split into train/validation data.

## 3. Inference for any champion

Open `inference.ipynb` and use:

```python
PROJECT_DIR = "/content/drive/MyDrive/text-orientation"
ARTIFACT_SOURCE = "drive"
BATCH_SIZE = None
NUM_WORKERS = 2
RESUME_INFERENCE = True
SMOKE_IMAGES = None
RUN_TESTS = False
```

Choose exactly one selector:

```python
MODEL = "best"                         # best standard single model
MODEL = "mobilenet_v3_large"           # standard MobileNet
MODEL = "efficientnet_b0"              # standard EfficientNet
MODEL = "vit_b_16"                     # standard ViT
MODEL = "mobilenet_v3_large_robust"    # robust MobileNet
MODEL = "vit_b_16_robust"             # robust ViT
MODEL = "best_ensemble"                # promoted ensemble
```

For a fast technical check set `SMOKE_IMAGES=64`; it does not create a valid
20,000-row submission. For the actual result restore `SMOKE_IMAGES=None`.
Outputs are written below `PROJECT_DIR/inference/runs/`; the final CSV is the
`submission.csv` in the latest run directory. `best_ensemble` requires the
complete Drive registry and therefore cannot use `ARTIFACT_SOURCE="upload"`.

## 4. Standard model training

Open `train.ipynb`. First run a cheap pipeline check:

```python
MODEL = "mobilenet"       # or "efficientnet", "vit"
QUICK_RUN = True
PROMOTE_CHAMPION = False
RUN_CALIBRATION = False
RESUME_TRAINING = True
RUN_TESTS = False
TRAIN_BATCH_SIZE = None
VALIDATION_BATCH_SIZE = None
```

Then run the selected architecture on the full synthetic budget:

```python
QUICK_RUN = False
PROMOTE_CHAMPION = True
RUN_CALIBRATION = True
RESUME_TRAINING = True
```

Architecture choices are `mobilenet`, `efficientnet`, and `vit`. Leave batch
sizes at `None` to use config defaults. If ViT runs out of memory, set
`TRAIN_BATCH_SIZE=8` and `VALIDATION_BATCH_SIZE=16`. Recovery state is saved
after epochs under `training/recovery/<model>/<quick|full>/last.pt`; rerunning
with `RESUME_TRAINING=True` continues from it. Only a full calibrated run can
update its architecture-specific champion in `registry/champions/`.

## 5. Robust fine-tuning

`robust_train.ipynb` starts from an existing standard champion. It supports
MobileNet and ViT, not EfficientNet.

Fast check:

```python
MODEL = "vit"                    # or "mobilenet"
QUICK_RUN = True
AUGMENTATION_PROFILE = "robust"
EPOCHS = 3
RESUME_TRAINING = True
PROMOTE_ROBUST_CHAMPION = False
```

Full run:

```python
QUICK_RUN = False
EPOCHS = 3
PROMOTE_ROBUST_CHAMPION = True
```

Defaults are ViT `16/32` and MobileNet `64/128` for train/validation batch
sizes. Robust models are isolated under `registry/robust`; they never replace
the standard champions. The unchanged validation profile makes standard and
robust metrics comparable.

## 6. Ensemble search

Open `ensemble.ipynb` only after every candidate directory contains
`best.pt`, `calibration.json`, and `validation_predictions.npz`. Set the
candidate selectors and their full run directories, for example:

```python
CANDIDATES = {
    "mobilenet_v3_large_robust": f"{PROJECT_DIR}/training/runs/robust/mobilenet_v3_large/full",
    "vit_b_16_robust": f"{PROJECT_DIR}/training/runs/robust/vit_b_16/full",
}
WEIGHT_STEP = 0.05
PROMOTE_ENSEMBLE = True
```

The search uses aligned validation probabilities and refuses incompatible
protocols or target order. It does not retrain networks. The report is written
to `evaluation/ensembles/search.json`; the promoted manifest is stored under
`registry/ensembles/` and becomes available as `MODEL="best_ensemble"`.

## 7. Restricted Optuna search

Open `optuna.ipynb`. Start with smoke mode; then use the deadline-sized search:

```python
SMOKE_RUN = True       # first technical check
N_TRIALS = 2
```

For the real restricted search:

```python
SMOKE_RUN = False
N_TRIALS = 8           # 8–12 is the intended range
FINETUNE_EPOCHS = 3
TRAIN_BATCH_SIZE = 64
VALIDATION_BATCH_SIZE = 128
```

The study is resumable and isolated under `tuning/optuna/`. A best Optuna
trial is not automatically a production champion: its exported config must be
confirmed with a full training and calibration run before promotion.

## Common failures

- `ModuleNotFoundError: src`: run the notebook setup cell, which changes into
  the cloned repository and invokes scripts as Python modules.
- Missing `registry/leaderboard.json`: `PROJECT_DIR` points to the wrong Drive
  folder or the registry has not been copied there.
- CUDA out of memory on ViT: use batch sizes `8/16`.
- Interrupted run: keep the same `PROJECT_DIR`, model, mode, and
  `RESUME_TRAINING=True`.
- Tests fail during an urgent GPU run: set `RUN_TESTS=False`; run tests
  separately in a clean runtime rather than hiding a training error.
