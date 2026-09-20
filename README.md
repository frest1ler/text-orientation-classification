# text-orientation-classification
Avito Bootcamp test task: text orientation classification.

## Final solution

The submitted solution classifies whether text in an image is upright (`0`) or
rotated by 180 degrees (`1`). It does not perform OCR. The final prediction is
a calibrated probability `p_180` produced by a probability ensemble:

- 75% robust ViT-B/16;
- 25% robust MobileNetV3-Large.

Both models are trained on deterministic synthetic upright/rotated pairs. The
robust profile adds bounded perspective, crop, blur, downscale, shadow/glare,
and occlusion only to the training split. Validation remains unchanged, which
makes the standard/robust comparison valid. At inference each image is
evaluated together with its exact 180-degree rotation, then the paired
probabilities are combined symmetrically and calibrated using five-fold OOF
temperature scaling.

The reviewer entry point is
[`solution.ipynb`](notebooks/colab/solution.ipynb). It downloads the immutable
artifact bundle, accepts the provided `test.zip`, verifies hashes, runs the two
components sequentially to stay within Colab memory, and creates the submitted
`submission.csv`.

### Reproduce the submitted result in Google Colab

This is the shortest path for a reviewer. No training and no Google Drive
registry are required.

1. Open [`notebooks/colab/solution.ipynb`](notebooks/colab/solution.ipynb) in
   Google Colab and select a GPU runtime.
2. Keep the pinned `REVISION`, `ARTIFACT_URL`, and `ARTIFACT_SHA256` unchanged.
3. Put the issued archive at `/content/test.zip`, or leave it absent and select
   `test.zip` in the upload dialog created by the notebook.
4. Choose **Runtime → Run all**. The notebook clones the pinned source revision,
   installs dependencies, downloads the immutable Release asset, verifies its
   SHA-256, and runs the final ensemble.
5. Download the generated `submission.csv`. A full run must contain 20,000
   unique IDs. For the original test archive its expected SHA-256 is
   `cf3be491f4df56015bce7f78f5db9c35ea7ad5f651d6fdf801ebff99fd1a5482`.

The model bundle is published separately as a GitHub Release asset and is not
stored in Git history:
[`text-orientation-solution-artifacts.zip`](https://github.com/frest1ler/text-orientation-classification/releases/download/solution-v1/text-orientation-solution-artifacts.zip).

#### Alternative: automatically download `test.zip` from Google Drive

The submitted `solution-v1` notebook normally opens a local file-upload dialog
when `/content/test.zip` is absent. To download the organizer-provided test
archive from this
[Google Drive folder](https://drive.google.com/drive/folders/16oPfpaQ9UQcHhrUCBcumce58e76sM0Au?usp=drive_link)
instead, replace the notebook's entire data-and-artifact loading cell (the cell
that starts with `test_zip = Path(TEST_ZIP_PATH)`) with the cell below. Then use
**Runtime → Run all** as usual.

```python
import hashlib
import shutil
import zipfile
from pathlib import Path

import gdown

TEST_DATA_DRIVE_URL = "https://drive.google.com/drive/folders/16oPfpaQ9UQcHhrUCBcumce58e76sM0Au?usp=drive_link"
TEST_DATA_DOWNLOAD_DIR = Path("/content/avito-test-data")

# Prefer an archive already placed at TEST_ZIP_PATH. Otherwise download the
# shared Drive folder and locate exactly one test.zip inside it.
test_zip = Path(TEST_ZIP_PATH)
if not test_zip.is_file():
    TEST_DATA_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(
        url=TEST_DATA_DRIVE_URL,
        output=str(TEST_DATA_DOWNLOAD_DIR),
        quiet=False,
        use_cookies=False,
        remaining_ok=True,
    )
    test_archives = sorted(TEST_DATA_DOWNLOAD_DIR.rglob("test.zip"))
    if len(test_archives) != 1:
        raise RuntimeError(
            "Expected exactly one test.zip after the Google Drive download, "
            f"found {len(test_archives)}: {test_archives}"
        )
    test_zip = test_archives[0]

# Download the immutable model bundle from the GitHub Release and verify it.
bundle = Path("/content/text-orientation-solution-artifacts.zip")
if not bundle.is_file():
    gdown.download(ARTIFACT_URL, str(bundle), quiet=False, fuzzy=True)

digest_builder = hashlib.sha256()
with bundle.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest_builder.update(chunk)
digest = digest_builder.hexdigest()
if digest != ARTIFACT_SHA256:
    raise ValueError(f"Artifact SHA-256 mismatch: {digest}")

# Prepare the project consumed by the regular inference cell.
project = Path(PROJECT_DIR)
(project / "data").mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(bundle) as archive:
    archive.extractall(project)

destination = project / "data/test.zip"
if destination.resolve() != test_zip.resolve():
    shutil.copy2(test_zip, destination)

print({
    "test_zip": str(destination),
    "artifact_sha256": digest,
})
```

`TEST_ZIP_PATH` always denotes a local filesystem path; a Drive folder URL
cannot be assigned to it directly. The replacement cell converts the public
folder URL into a local archive automatically. The Drive folder must allow
read access to anyone with the link. The test data remains separate from the
GitHub model Release.

### Run a particular trained model

Use [`notebooks/colab/inference.ipynb`](notebooks/colab/inference.ipynb) when a
persistent registry already exists at
`/content/drive/MyDrive/text-orientation`. Put the test archive at
`text-orientation/data/test.zip`, then set `MODEL` to one of:

| `MODEL` | Result |
| --- | --- |
| `best` | Best standard single-model champion |
| `mobilenet_v3_large` | Standard MobileNetV3-Large |
| `efficientnet_b0` | Standard EfficientNet-B0 |
| `vit_b_16` | Standard ViT-B/16 |
| `mobilenet_v3_large_robust` | Robust MobileNetV3-Large |
| `vit_b_16_robust` | Robust ViT-B/16 |
| `best_ensemble` | Promoted ensemble from `registry/ensembles` |

For a complete submission use `SMOKE_IMAGES=None`; use `SMOKE_IMAGES=64` only
to verify the pipeline. Keep `BATCH_SIZE=None` for architecture-safe defaults.
Exact folder layouts, copy/paste settings, expected outputs, training,
recovery, robust fine-tuning, ensemble search, and Optuna are documented in
[`notebooks/colab/README.md`](notebooks/colab/README.md). Rented-server
instructions are in
[`notebooks/cloudcompute/README.md`](notebooks/cloudcompute/README.md).

### Results

| Candidate | Validation 1 − Brier | Hidden-test 1 − Brier |
| --- | ---: | ---: |
| MobileNetV3-Large | 0.927589 | 0.855727 |
| Robust MobileNetV3-Large | 0.934886 | 0.875987 |
| ViT-B/16 | 0.943337 | 0.905859 |
| Robust ViT-B/16 | 0.954320 (OOF calibrated) | 0.928290 |
| 25% robust MobileNet + 75% robust ViT | **0.956555** | **0.929383** |

The final hidden-test Brier error is `0.07061748`. The submitted file contains
20,000 unique image IDs and has SHA-256
`cf3be491f4df56015bce7f78f5db9c35ea7ad5f651d6fdf801ebff99fd1a5482`.

An eight-trial, deadline-sized Optuna search was also run for MobileNet. Its
best reduced-budget trial reached validation Brier `0.132075` with learning
rate `1.9057e-4`, dropout `0.1545`, weight decay `5.47e-6`, and symmetry weight
`0.0550`. It was not promoted because a full-data confirmation run was not
completed; this avoids presenting tuning-only metrics as a final result.

## Data audit

The test archive is treated as immutable and read directly with Python's
`zipfile` module. Run the reproducible integrity and metadata audit with:

```bash
PYTHONPATH=. python3 -m src.data_audit \
  --zip-path test.zip \
  --output-dir artifacts/data_audit
```

The archive itself, per-image metadata, and contact sheet are ignored by Git.
The compact aggregate report is stored in
`artifacts/data_audit/summary.json`. See [docs/data_audit.md](docs/data_audit.md)
for findings from the initial audit.

Run the current tests with:

```bash
PYTHONPATH=. python3 -m pytest -q
```

Validate the source package and both Colab notebooks before delivery:

```bash
PYTHONPATH=. python3 -m scripts.verify_delivery
```

With a prepared persistent project, add `--project-dir /path/to/text-orientation`
to verify the test ZIP, registry, every champion hash, and best-model selection.
The complete hand-off procedure is in
[docs/delivery_checklist.md](docs/delivery_checklist.md).

## Configuration smoke check

The experiment is configured through a strict, validated YAML file. Check the
baseline configuration and current runtime device before starting expensive
work:

```bash
PYTHONPATH=. python3 scripts/check_config.py --config configs/baseline.yaml
```

Unknown keys and invalid values fail immediately. The baseline currently uses
a `384x96` MobileNetV3-Large setup; this is an experiment candidate, not a
preselected final model or resolution.

The implementation is developed and tested locally on CPU; training will use
an automatically detected GPU in Google Colab. See
[docs/project_structure.md](docs/project_structure.md) for the current module
boundaries and reproducibility policy.

## Synthetic data preview

Generate balanced, deterministic `0°/180°` pairs without writing a training
dataset to disk:

```bash
PYTHONPATH=. python3 scripts/preview_synthetic.py \
  --config configs/baseline.yaml \
  --split train \
  --pairs 8
```

The preview is ignored by Git. The generator design, split isolation, geometry
matching, font sources, and licences are described in
[docs/synthetic_data.md](docs/synthetic_data.md).

## Paired pipeline checks

Verify that the small technical CNN can memorize a fixed paired dataset:

```bash
PYTHONPATH=. python3 scripts/sanity_overfit.py
```

Run a short end-to-end train, validation, checkpoint, and restore check:

```bash
PYTHONPATH=. python3 scripts/smoke_train.py
```

The pair objective and observed sanity results are documented in
[docs/training_pipeline.md](docs/training_pipeline.md). The ordered model and
augmentation comparisons are tracked in
[docs/experiment_plan.md](docs/experiment_plan.md).

## Compact model training

Train the primary MobileNetV3-Large experiment:

```bash
PYTHONPATH=. python3 scripts/train.py --config configs/baseline.yaml
```

Run the controlled EfficientNet-B0 comparison:

```bash
PYTHONPATH=. python3 scripts/train.py --config configs/efficientnet_b0.yaml
```

The rectangular ViT-B/16 foundation and configuration are also available:

```bash
PYTHONPATH=. python3 scripts/train.py --config configs/vit_b_16.yaml
```

Its `384×96` positional-embedding adaptation is documented in
[docs/vit_b_16.md](docs/vit_b_16.md). Training, recovery, registry, inference,
and Colab wiring are implemented and evaluated in the final comparison above.

After full candidates have been promoted, create a protocol-safe comparison:

```bash
python3 -m scripts.compare_champions \
  --project-dir /path/to/text-orientation \
  --require-model mobilenet_v3_large \
  --require-model efficientnet_b0 \
  --require-model vit_b_16
```

The command refuses to rank models trained under different validation
protocols and writes JSON, CSV, and Markdown reports to
`evaluation/champions/`.

Both use torchvision ImageNet weights, rectangular preprocessing, paired
training, mixed precision on CUDA, and best-checkpoint selection by symmetric
Brier score. Details are in
[docs/compact_models.md](docs/compact_models.md).

For the first GPU verification, open `notebooks/colab/train.ipynb`, choose `mobilenet`,
`efficientnet`, or `vit`, keep `QUICK_RUN=True`, and run all cells on a T4
runtime.
The exact steps and the result archive to return are documented in
[docs/colab_run.md](docs/colab_run.md).

The same four workflows are available for a rented CloudCompute instance in
`notebooks/cloudcompute/`. They use `/root/text-orientation-state` for local
data, recovery checkpoints, registries, and exports and contain no Colab or
Google Drive dependency. See [notebooks/README.md](notebooks/README.md) for the
platform matrix and the path used to resume an interrupted ViT run.

Validation-selected probability ensembles are supported through
`notebooks/colab/ensemble.ipynb` and its CloudCompute counterpart. Ensemble
manifests live in an isolated `registry/ensembles/` branch; use
`MODEL="best_ensemble"` in the regular inference notebook. Components run
sequentially and retain independent recovery state. See
[docs/ensembles.md](docs/ensembles.md).

Full Colab runs are compared with a separate Google Drive champion for each
architecture. Checkpoints are promoted by symmetric Brier score only when the
validation protocol fingerprint matches; quick runs are archived without
promotion.

## Restricted Optuna search

Open `notebooks/colab/optuna.ipynb` for an isolated, resumable MobileNetV3-Large search.
It tunes only fine-tuning learning rate, weight decay, dropout, and symmetry
loss weight, using symmetric validation Brier as the objective. Start with the
two-trial smoke mode, then use the deadline-sized 8–12 trial search. Results
are stored below `tuning/optuna/`; the model registry is never modified by a
trial. The exported `best_config.yaml` must complete full training and
calibration before normal champion promotion can consider it. See
[docs/optuna.md](docs/optuna.md).

## Optional robust augmentation

`notebooks/colab/robust_train.ipynb` compares the unchanged `standard` train profile
with an isolated `robust` profile containing bounded perspective, cropping,
motion/downscale/spatial blur, shadow/glare, and occlusion. Robust train
degradations change reproducibly by epoch; validation remains unchanged.
Artifacts and recovery live below profile-specific directories, and full
robust champions use the separate `registry/robust/` branch. They are selected
explicitly with `MODEL="mobilenet_v3_large_robust"` or
`MODEL="vit_b_16_robust"`; `MODEL="best"` retains its
standard-only meaning. See [docs/robust_augmentation.md](docs/robust_augmentation.md).

## Calibration and OCR baselines

Evaluate leakage-resistant five-fold calibration for a completed run:

```bash
python3 -m scripts.calibrate \
  --run-dir artifacts/experiments/<run-name> \
  --config configs/baseline.yaml
```

This stores OOF validation predictions and a final calibrator fitted only
after method selection. Raw champion metrics remain unchanged.

The optional offline OCR baseline compares Tesseract confidence for the image
and its exact 180-degree rotation:

```bash
sudo apt-get install tesseract-ocr tesseract-ocr-rus
python3 -m scripts.ocr_baseline --config configs/baseline.yaml
```

OCR is deliberately separate from CNN training and makes no network/API calls.

## Progress and recovery

Training displays per-batch `tqdm` progress for both train and validation.
To persist an epoch-boundary recovery checkpoint, use:

```bash
python3 -m scripts.train \
  --config configs/baseline.yaml \
  --recovery-dir /path/to/project/training/recovery/mobilenet_v3_large/full \
  --resume
```

`last.pt` contains the model, optimizer, phase, epoch, early-stopping state,
history, and random-generator states. `best.pt` is mirrored separately.
Resume is refused if the configuration, runtime parameters, or critical
training source files differ.

The Colab notebook stores every persistent artifact below one `PROJECT_DIR`
and isolates recovery by architecture and run mode:
`training/recovery/<model>/quick` and `training/recovery/<model>/full`. This
prevents a quick smoke checkpoint from blocking or resuming a full run.

The canonical persistent layout is:

```text
text-orientation/
  data/
  registry/
  training/
  inference/
  ocr/
```

See [docs/project_layout.md](docs/project_layout.md) for the complete layout
and the non-destructive command that imports an existing full run ZIP into the
self-contained registry.

## Resumable test inference

Generate ordered direct, rotated, symmetric, and calibrated predictions from
the best registered champion:

```bash
python3 -m scripts.infer \
  --project-dir /path/to/text-orientation \
  --model best
```

The command validates the complete bundle, reads images directly from
`data/test.zip`, and resumes through fingerprinted atomic state. A completed
full run writes `submission.csv`, `predictions.csv`, `inference_report.json`,
and four contact sheets below `inference/runs/<run-id>/`. A limited
`--limit 64` smoke run writes diagnostics but deliberately does not create a
submission.

For a reviewer-friendly Colab launch, open `notebooks/colab/inference.ipynb`, choose
`MODEL="best"` or a named champion, and run all cells. It supports both the
Drive registry and a separately uploaded champion bundle. See
[docs/inference_pipeline.md](docs/inference_pipeline.md).
