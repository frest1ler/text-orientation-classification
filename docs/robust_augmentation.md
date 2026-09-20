# Optional robust training profile

The robust experiment is intentionally isolated from standard champions. It
fine-tunes the selected MobileNetV3-Large or ViT-B/16 champion while leaving the original
checkpoint, recovery state, leaderboard, and submission untouched.

## Profiles

`standard` reproduces the existing deterministic train renderer. `robust`
adds bounded rotation, perspective, edge crop, motion/downscale/spatial blur, local
shadow or glare, and small occlusion. No more than two added operations are
used for one image. The transformed upright image is created first and its
paired input is always the exact 180-degree rotation, preserving the task
label and symmetry contract.

Only train data is epoch-varying. Seeds are derived from
`(global_seed, epoch, sample_id)`. Validation continues to use the unchanged
`PairedSyntheticDataset` and therefore remains directly comparable with the
existing champions.

## Storage isolation

```text
text-orientation/
  training/runs/<standard|robust>/<model>/<quick|full>/
  training/recovery/<standard|robust>/<model>/<quick|full>/
  registry/robust/
    leaderboard.json
    champions/<model>/
```

The robust promotion command refuses standard, quick, partial-validation, or
untraceable runs. The source checkpoint path, SHA-256, and epoch are recorded
in `runtime.json`.

## Colab experiment

Open `notebooks/colab/robust_train.ipynb`. Start with:

```python
MODEL = "vit"  # vit | mobilenet
AUGMENTATION_PROFILE = "robust"
QUICK_RUN = True
PROMOTE_ROBUST_CHAMPION = False
```

After inspecting the quick history, use `QUICK_RUN=False` for the controlled
50,000/5,000 experiment. Full runs use three fine-tuning epochs and learning
rate `1e-5` for ViT or `2e-5` for MobileNet by default. Architecture-specific
config, batch sizes, checkpoint, recovery directory, calibration, comparison,
and export paths are selected together. Calibration always uses the unchanged
validation set.

The robust candidate is explicit during inference:

```python
MODEL = "mobilenet_v3_large_robust"
# or
MODEL = "vit_b_16_robust"
```

`MODEL="best"` remains restricted to standard individual champions. Robust
quality is not claimed until the GPU experiment completes. The acceptance
gate is a validation Brier improvement of at least `0.005`; otherwise the
corresponding standard model remains the chosen model.

After a full promotion the notebook writes the protocol-checked decision to
`evaluation/robust/<model>/comparison.json`. Registration records the isolated robust
candidate even when it is worse; this report determines whether it is
recommended for final inference.
