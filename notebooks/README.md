# Notebooks

The notebooks are thin platform wrappers around the same tested Python scripts.
Model, validation, recovery, registry, and submission logic therefore stays
identical between environments.

| Task | Google Colab | CloudCompute |
| --- | --- | --- |
| Train MobileNet, EfficientNet, or ViT | `colab/train.ipynb` | `cloudcompute/train.ipynb` |
| Test inference and submission | `colab/inference.ipynb` | `cloudcompute/inference.ipynb` |
| Optuna search | `colab/optuna.ipynb` | `cloudcompute/optuna.ipynb` |
| Robust fine-tuning | `colab/robust_train.ipynb` | `cloudcompute/robust_train.ipynb` |

Colab uses the mounted Google Drive project directory. CloudCompute uses two
local paths by default:

- `/root/text-orientation-classification` — cloned source repository;
- `/root/text-orientation-state` — data, registry, recovery checkpoints, and
  exported results that must survive notebook restarts.

For resuming a migrated run, copy its recovery directory before starting. For
example, a full ViT run belongs in
`/root/text-orientation-state/training/recovery/vit_b_16/full` and must contain
`last.pt`. Stop the instance to preserve its disk; download results before
deleting the instance.
