# Project structure

The project is a new implementation tailored to binary text-orientation
classification. It does not reuse code, models, configuration, or checkpoints
from `vision_classifier_pipeline`.

Persistent Drive artifacts follow the portable layout documented in
[project_layout.md](project_layout.md).

## Current modules

```text
colab_train.ipynb        Concise Colab GPU training and artifact export
colab_inference.ipynb    Reviewer-facing best/named champion inference
configs/
  baseline.yaml          Validated baseline experiment settings
  efficientnet_b0.yaml   Controlled compact-backbone comparison
scripts/
  calibrate.py           OOF calibration evaluation and final fit
  import_champion.py     Import a run ZIP into the unified model registry
  infer.py               Resumable symmetric test prediction CLI
  verify_delivery.py     Read-only source and persistent-artifact readiness gate
  check_config.py        Configuration and device smoke check
  preview_synthetic.py   Local visual check of generated pairs
  sanity_overfit.py      Tiny-set paired learning verification
  smoke_train.py         End-to-end train/checkpoint smoke run
  train.py               Frozen-head and fine-tuning training entry point
  promote_champion.py    Per-model Drive champion promotion
  ocr_baseline.py        Offline two-orientation Tesseract baseline
src/
  calibration.py         Temperature/Platt fitting and OOF selection
  champions.py           Comparable-run fingerprint and champion registry
  config.py              Typed YAML loading and strict validation
  data_audit.py          Immutable ZIP audit from stage 1
  metrics.py             Brier score and diagnostic probability metrics
  models.py              Small CNN and compact pretrained backbones
  inference.py           Champion loading, test loader, and symmetric batches
  inference_recovery.py  Atomic ordered prediction recovery
  diagnostics.py         Numeric inference report and visual contact sheets
  ocr.py                 Tesseract TSV confidence and orientation score
  project_layout.py      Canonical persistent project paths and artifact sources
  registry.py            Validated best/named champion bundle selection
  recovery.py            Atomic epoch checkpoints and compatibility fingerprint
  readiness.py           Notebook, config, ZIP, and registry delivery checks
  test_data.py            Ordered image access directly from test ZIP
  submission.py          Strict template-preserving submission construction
  reproducibility.py     Seeds, DataLoader generator, device selection
  synthetic.py           Deterministic rendering and paired dataset
  text_corpus.py         Split-specific self-contained text sources
  training.py            Single and paired train/evaluation loops
  transforms.py          Aspect-preserving resize and padding
tests/
  test_calibration.py
  test_config.py
  test_data_audit.py
  test_inference.py
  test_inference_recovery.py
  test_ocr.py
  test_registry.py
  test_recovery.py
  test_test_data.py
  test_reproducibility.py
  test_synthetic.py
  test_training_pipeline.py
  test_readiness.py
  test_reviewer_flow.py
  test_submission.py
  test_diagnostics.py
```

## Configuration policy

`configs/baseline.yaml` is the single source of experiment settings. The
loader rejects missing sections, unknown keys, unsupported models and
languages, invalid probabilities, invalid learning rates, and image sizes not
divisible by 16. A serialisable snapshot is available through
`Config.to_dict()` for storing alongside each future checkpoint.

The initial `384x96` resolution and MobileNetV3-Large model are baseline
experiment candidates. They are not final decisions and will be compared by
validation Brier score.

## Reproducibility policy

`seed_everything()` covers Python, NumPy, PyTorch CPU/CUDA, cuDNN flags, and
deterministic PyTorch algorithms. DataLoaders must receive both
`seed_worker()` and `make_generator()` once datasets are implemented.

The local development environment intentionally runs on CPU because its
installed PyTorch CUDA build is newer than the host NVIDIA driver. Training is
planned for a compatible Google Colab GPU runtime. Device selection remains
automatic and never depends on a specific GPU model.
