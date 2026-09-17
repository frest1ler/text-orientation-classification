# Synthetic training data

## Design

No labelled competition training set is available. Training data is generated
on demand as balanced pairs from an upright synthetic base image:

```text
base image -> unchanged -> label 0
           -> rotate 180 degrees -> label 1
```

The base image is determined only by `(global_seed, split, sample_id)`. It can
therefore be regenerated in any order and with any number of DataLoader
workers. Both orientations always remain in the same split.

## Split isolation

The split occurs before orientation pairs are created:

- Train fonts: DejaVu Sans/Serif and Noto Sans/Serif, regular and bold.
- Validation fonts: Liberation Sans/Serif, regular and bold.
- Russian and English word lists are disjoint between train and validation.
- Structured numeric ranges differ between train and validation.
- Split identity is part of the per-sample random seed.

This is intentionally stricter than a random image-level split and prevents a
font file or base rendering from appearing in both subsets.

## Text and visual variation

The self-contained text corpus contains manually curated generic words rather
than copied candidate solutions or test labels. It generates Russian, English,
numeric, and mixed-script strings including prices, dates, identifiers, and
phone-like patterns. Case, separators, word count, and optional line breaks
vary deterministically.

Rendering varies font, size, weight, text/background colour, padding,
placement, solid/gradient/noisy backgrounds, brightness, contrast, mild blur,
noise, small-angle rotation, and JPEG degradation. No horizontal flip or
unlabelled 180-degree augmentation is used.

## Geometry matching

Only aggregate, unlabelled geometry from the test audit is used. Width, height,
and aspect-ratio quantiles are stored in `configs/baseline.yaml`; the generator
does not read test images or per-image metadata. Text is rendered into an alpha
layer and fitted onto a sampled canvas without changing its aspect ratio.

Measured over 5,000 examples per split:

| Metric | Test | Synthetic train | Synthetic validation |
|---|---:|---:|---:|
| Median aspect ratio | 4.89 | 4.82 | 4.90 |
| 95th percentile aspect ratio | 14.45 | 13.94 | 14.18 |
| Share with aspect ratio > 8 | 22.07% | 22.62% | 23.00% |
| Maximum width | 1,599 | 1,599 | 1,599 |

These figures validate geometry only. Model selection must still use held-out
Brier score.

## Fonts and licences

Exact font binaries are stored in `assets/fonts/` so rendering does not depend
on device-specific system packages. `manifest.json` records every SHA-256 and
is verified before generation.

- DejaVu Fonts: Bitstream Vera licence with DejaVu changes in the public
  domain. Full package notice: `assets/fonts/licenses/DejaVu.txt`.
- Noto Fonts: SIL Open Font License 1.1. Full package notice:
  `assets/fonts/licenses/Noto.txt`.
- Liberation Fonts: SIL Open Font License 1.1. Full package notice:
  `assets/fonts/licenses/Liberation.txt`.

Ubuntu Fonts were deliberately excluded.

## Cross-device reproducibility

NumPy and Pillow are pinned because RNG and rasterisation changes can alter
pixels. `assets/fonts/rendering_environment.json` records the reference
FreeType version. Snapshot tests cover six rendered examples and fail visibly
if another environment produces different pixels.

Exact GPU training can still vary between hardware and CUDA versions. The
submitted predictions will therefore be reproduced from a saved checkpoint
with deterministic inference, while training reproducibility remains
best-effort within PyTorch's deterministic guarantees.
