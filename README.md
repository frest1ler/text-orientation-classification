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

Both use torchvision ImageNet weights, rectangular preprocessing, paired
training, mixed precision on CUDA, and best-checkpoint selection by symmetric
Brier score. Details are in
[docs/compact_models.md](docs/compact_models.md).

For the first GPU verification, open `colab_train.ipynb`, keep
`MODEL="mobilenet"` and `QUICK_RUN=True`, and run all cells on a T4 runtime.
The exact steps and the result archive to return are documented in
[docs/colab_run.md](docs/colab_run.md).
