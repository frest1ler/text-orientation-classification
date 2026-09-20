# Notebooks

The notebooks are thin platform wrappers around the same tested Python scripts.
Model, validation, recovery, registry, and submission logic therefore stays
identical between environments.

| Task | Google Colab | CloudCompute |
| --- | --- | --- |
| Train MobileNet, EfficientNet, or ViT | `colab/train.ipynb` | `cloudcompute/train.ipynb` |
| Test inference and submission | `colab/inference.ipynb` | `cloudcompute/inference.ipynb` |
| Optuna search | `colab/optuna.ipynb` | `cloudcompute/optuna.ipynb` |
| Robust MobileNet/ViT fine-tuning | `colab/robust_train.ipynb` | `cloudcompute/robust_train.ipynb` |
| Ensemble weight search | `colab/ensemble.ipynb` | `cloudcompute/ensemble.ipynb` |
| Reproduce final submission | `colab/solution.ipynb` | — |

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

Both robust notebooks use one `MODEL` setting: `"mobilenet"` or `"vit"`.
It selects the matching standard champion, config, safe batch defaults,
recovery namespace, calibration command, robust registry entry, and comparison
report. A full promoted ViT candidate is available to inference explicitly as
`MODEL="vit_b_16_robust"`; it never replaces `MODEL="best"`.

Ensemble search consumes each champion run's `validation_predictions.npz`,
checks the shared validation protocol and target order, and stores only a
small manifest below `registry/ensembles/`. Inference with
`MODEL="best_ensemble"` runs components sequentially, so only one model is in
GPU memory at a time.
