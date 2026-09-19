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

## Still pending

The model, configuration, pretrained-weight adaptation, direct training CLI,
calibration loader, and inference loader are now compatible. The remaining
ViT stages are Colab wiring, explicit recovery/champion integration checks,
checkpoint/inference tests, a quick T4 run, and only then a full candidate run.
No ViT quality result is claimed before those GPU experiments complete.
