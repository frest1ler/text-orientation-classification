# Paired training pipeline

## Purpose of the stage

The small CNN is a technical baseline used to verify labels, preprocessing,
probability metrics, gradients, checkpoints, and paired inference. It is not a
candidate final architecture.

## Pair construction

`PairedSyntheticDataset` returns one logical sample containing:

```text
image       orientation chosen deterministically from sample index
rotated     exact 180-degree counterpart
target      orientation label of image
```

The target of `rotated` is always `1 - target`. Half of the dataset places the
upright version first and half places the rotated version first. Both tensors
are sent through the same model weights in one concatenated batch.

## Objective

For logits `z(x)` and `z(R(x))`, training minimises:

```text
0.5 * [BCE(z(x), y) + BCE(z(R(x)), 1-y)]
+ lambda * mean((sigmoid(z(x)) + sigmoid(z(R(x))) - 1)^2)
```

The configured initial symmetry weight is `0.1`; it remains an experimental
value and must later be compared with `0.0` by validation Brier score.

The symmetry-enforced probability is:

```text
p = 0.5 * [sigmoid(z(x)) + 1 - sigmoid(z(R(x)))]
```

Direct and symmetric validation metrics are both recorded.

## Sanity overfit

The paired small CNN memorised eight fixed training pairs at epoch 28:

```text
symmetric Brier score: 0.01999
symmetric accuracy:    1.00000
symmetric ROC-AUC:     1.00000
mean symmetry error:   0.03819
```

This demonstrates that labels, pair direction, BCE targets, gradients, and
symmetrisation are connected correctly. GroupNorm is used instead of
BatchNorm so train/evaluation behaviour remains stable for small batches.

## Smoke validation

A deliberately tiny two-epoch CPU run with 128 train pairs and 32 held-out
validation pairs remained close to random. This is not treated as a model
quality result: validation uses disjoint words and Liberation fonts while the
run sees only 128 DejaVu/Noto examples. The run verified that the best
checkpoint is selected by symmetric Brier score and that loading it reproduces
the metrics exactly.

Use:

```bash
PYTHONPATH=. python3 scripts/sanity_overfit.py
PYTHONPATH=. python3 scripts/smoke_train.py
```
