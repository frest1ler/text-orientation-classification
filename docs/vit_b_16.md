# Rectangular ViT-B/16

## Implemented foundation (stages 1–5)

The candidate uses torchvision ViT-B/16 and the same binary `[batch]` logit
contract as the CNN models. Its binary head contains one output and works with
the existing BCE and symmetric paired objective.

The input remains `384×96` rather than being distorted into a square. With a
16-pixel patch this creates a `24×6` grid: 144 image tokens and one unchanged
CLS token. Torchvision's stock ViT accepts only square inputs, so the project
uses a small subclass that changes only fixed-size input validation and patch
grid construction; encoder blocks remain torchvision implementations.

For ImageNet weights, the original square positional embedding is split into
CLS and image positions. Image positions are reshaped to their square grid,
bicubically interpolated to `6×24`, and flattened in the same order as the
convolutional patch projection. The CLS position is copied bit-for-bit. All
other pretrained parameters are loaded strictly before replacing the
1,000-class head with the binary head.

`configs/vit_b_16.yaml` uses:

- input `384×96`;
- train batch size 16 and validation/inference batch size 32;
- frozen learning rate `1e-3`;
- fine-tuning learning rate `3e-5`;
- the same seed, synthetic samples, validation set, symmetry loss, and Brier
  selection criterion as the compact candidates.

Frozen training leaves only 769 head parameters trainable. Fine-tuning uses
the existing complete-model unfreeze operation.

## Training, registry, and inference (stages 6–10)

The standard training phases, AMP, symmetry loss, epoch recovery, calibration,
and early stopping are shared with the CNN candidates. Recovery is isolated at
`training/recovery/vit_b_16/<quick|full>`.

`colab_train.ipynb` accepts `MODEL="vit"`. A quick run uses batch sizes 16/32;
full runs read the same values from the ViT config. CUDA OOM is reported with
an explicit 8/16 fallback suggestion but batch size is never changed silently.
A calibrated full run may promote only `registry/champions/vit_b_16/`, while
the global `best` selector continues to compare all independent champions by
the locked symmetric Brier protocol.

Inference reconstructs the rectangular dimensions from the champion
checkpoint. When no command-line batch override is supplied, it uses the
selected champion's own inference batch size, so `MODEL="best"` remains safe
if ViT wins. Direct/rotated symmetric prediction, calibration, recovery,
reports, contact sheets, and submission construction require no ViT-specific
branch.

Tests cover rectangular forward, freeze/unfreeze, positional interpolation,
strict config, independent registry promotion, champion reconstruction,
symmetric inference, and both notebook entry points. Generic checkpoint and
recovery round-trip tests exercise the shared serialization mechanism.

## Still pending

The implementation is ready for a quick T4 run. Only measured GPU memory,
speed, resume behavior, and validation metrics from that run can justify a
full ViT experiment. No ViT quality result is claimed before it completes.
