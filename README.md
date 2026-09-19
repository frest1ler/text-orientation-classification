# text-orientation-classification
Avito Bootcamp test task: text orientation classification.

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
[docs/vit_b_16.md](docs/vit_b_16.md). Colab wiring and the GPU smoke run remain
the next ViT stages; the model should not yet be treated as an evaluated
champion.

Both use torchvision ImageNet weights, rectangular preprocessing, paired
training, mixed precision on CUDA, and best-checkpoint selection by symmetric
Brier score. Details are in
[docs/compact_models.md](docs/compact_models.md).

For the first GPU verification, open `colab_train.ipynb`, keep
`MODEL="mobilenet"` and `QUICK_RUN=True`, and run all cells on a T4 runtime.
The exact steps and the result archive to return are documented in
[docs/colab_run.md](docs/colab_run.md).

Full Colab runs are compared with a separate Google Drive champion for each
architecture. Checkpoints are promoted by symmetric Brier score only when the
validation protocol fingerprint matches; quick runs are archived without
promotion.

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

For a reviewer-friendly Colab launch, open `colab_inference.ipynb`, choose
`MODEL="best"` or a named champion, and run all cells. It supports both the
Drive registry and a separately uploaded champion bundle. See
[docs/inference_pipeline.md](docs/inference_pipeline.md).
