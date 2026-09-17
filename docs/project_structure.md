# Project structure

The project is a new implementation tailored to binary text-orientation
classification. It does not reuse code, models, configuration, or checkpoints
from `vision_classifier_pipeline`.

## Current modules

```text
configs/
  baseline.yaml          Validated baseline experiment settings
scripts/
  check_config.py        Configuration and device smoke check
  preview_synthetic.py   Local visual check of generated pairs
src/
  config.py              Typed YAML loading and strict validation
  data_audit.py          Immutable ZIP audit from stage 1
  reproducibility.py     Seeds, DataLoader generator, device selection
  synthetic.py           Deterministic rendering and paired dataset
  text_corpus.py         Split-specific self-contained text sources
tests/
  test_config.py
  test_data_audit.py
  test_reproducibility.py
  test_synthetic.py
```

Modules for transforms, models, losses, training, calibration, inference, and
submission creation will be added only when their respective stages are
implemented. Empty placeholder modules are deliberately avoided.

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
