# Optional robust training profile

The robust experiment is intentionally isolated from standard champions. It
fine-tunes the current MobileNetV3-Large champion while leaving the original
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
  training/runs/<standard|robust>/mobilenet_v3_large/<quick|full>/
  training/recovery/<standard|robust>/mobilenet_v3_large/<quick|full>/
  registry/robust/
    leaderboard.json
    champions/mobilenet_v3_large/
```

The robust promotion command refuses standard, quick, partial-validation, or
untraceable runs. The source checkpoint path, SHA-256, and epoch are recorded
in `runtime.json`.

## Colab experiment

Open `notebooks/colab/robust_train.ipynb`. Start with:

```python
AUGMENTATION_PROFILE = "robust"
QUICK_RUN = True
PROMOTE_ROBUST_CHAMPION = False
```

After inspecting the quick history, use `QUICK_RUN=False` for the controlled
50,000/5,000 experiment. Full runs use three fine-tuning epochs and learning
rate `2e-5` by default. Calibration always uses the unchanged validation set.

The robust candidate is explicit during inference:

```python
MODEL = "mobilenet_v3_large_robust"
```

`MODEL="best"` remains restricted to standard individual champions. Robust
quality is not claimed until the GPU experiment completes. The acceptance
gate is a validation Brier improvement of at least `0.005`; otherwise the
standard MobileNet remains the chosen model.

After a full promotion the notebook writes the protocol-checked decision to
`evaluation/robust/comparison.json`. Registration records the isolated robust
candidate even when it is worse; this report determines whether it is
recommended for final inference.
