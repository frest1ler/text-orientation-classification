# Compact pretrained models

## Implemented architectures

The first model-quality experiments use two torchvision backbones:

| Model | Parameters with binary head | Role |
|---|---:|---|
| MobileNetV3-Large | 4,203,313 | Primary compact baseline |
| EfficientNet-B0 | approximately 4.0M | Controlled backbone comparison |

Both replace the original ImageNet classifier with a one-logit `BinaryLinear`
head. The head guarantees a common `[batch]` output contract across every
model. Both architectures accept the rectangular `384x96` input because their
feature extractors end in adaptive pooling.

Pretrained models use ImageNet mean and standard deviation after
aspect-ratio-preserving resize and padding. Images are never stretched to a
square or aggressively cropped.

## Training phases

Each experiment has two configured phases:

1. Freeze `model.features` and train the classifier head.
2. Unfreeze every parameter, create a new optimizer, and fine-tune with a
   smaller learning rate.

Early-stopping patience resets at the phase boundary, while the best global
symmetric Brier score remains shared. This prevents a plateau during frozen
training from prematurely ending full fine-tuning.

Each experiment stores:

```text
artifacts/experiments/<experiment>/
  best.pt       model, optimizer, epoch, metrics, config
  config.json   immutable YAML snapshot
  runtime.json  actual device and CLI overrides
  history.json  per-epoch losses and direct/symmetric metrics
```

The directory is excluded from Git.

## Commands

MobileNetV3-Large:

```bash
PYTHONPATH=. python3 scripts/train.py --config configs/baseline.yaml
```

EfficientNet-B0:

```bash
PYTHONPATH=. python3 scripts/train.py --config configs/efficientnet_b0.yaml
```

The first run may download official torchvision ImageNet weights. Subsequent
training and all final inference are local. Device selection is automatic;
mixed precision is enabled only when CUDA is available.

For restricted local environments that disallow DataLoader subprocesses:

```bash
PYTHONPATH=. python3 scripts/train.py \
  --config configs/baseline.yaml \
  --num-workers 0
```

`runtime.json` records this override. The committed Colab configurations keep
two workers.

## Local verification

Unit tests construct both backbones with `pretrained=False`, so tests never
depend on network access. A tiny CPU run also exercised frozen and fine-tuning
phases without pretrained weights. Its metrics are not a quality measurement;
real experiments require pretrained weights, the configured data volume, and
a Colab GPU.
