# Test data audit

## Scope

The audit reads `test.zip` directly and does not extract or modify its files.
It validates archive paths, decodes every image with Pillow, checks the sample
submission, records image geometry and basic pixel statistics, and searches
for exact and perceptual-hash duplicate candidates.

The audit is deterministic with seed `42`. Its generated contact sheet is for
unlabelled visual inspection only and is excluded from Git.

## Integrity results

- Archive size: 681,431,081 bytes.
- Archive members: 20,001 (20,000 images and `sample_submission.csv`).
- Successfully decoded images: 20,000 of 20,000.
- Image encoding: PNG; image mode: RGB.
- Broken images: 0.
- Exact SHA-256 duplicate groups: 0.
- Submission rows and unique IDs: 20,000.
- Submission columns: exactly `image_id,p_180`.
- Submission IDs and image IDs match exactly, including order.

The simple 64-bit difference hash produced four two-image collision groups.
They are only candidates, not confirmed duplicates. Their different SHA-256
hashes, dimensions, and file sizes show that they are not byte-identical.

## Geometry

| Statistic | Width | Height | Aspect ratio |
|---|---:|---:|---:|
| Minimum | 25 | 10 | 0.72 |
| 1st percentile | 40 | 12 | 1.76 |
| 5th percentile | 61 | 16 | 2.32 |
| Median | 238 | 42 | 4.89 |
| 95th percentile | 1,047 | 195 | 14.45 |
| 99th percentile | 1,406 | 289 | 22.61 |
| Maximum | 1,599 | 492 | 49.67 |

Additional observations:

- 97.70% of images have aspect ratio greater than 2.
- 65.02% have aspect ratio greater than 4.
- 22.07% have aspect ratio greater than 8.
- Only 6 images are square or portrait-oriented.
- 56.61% are no more than 48 pixels high.
- 87.19% are no more than 128 pixels high.

## Pixel and file statistics

| Statistic | File size (bytes) | Pixel mean | Pixel standard deviation |
|---|---:|---:|---:|
| Minimum | 582 | 5.75 | 2.56 |
| Median | 13,500 | 141.27 | 52.83 |
| 95th percentile | 138,699 | 219.50 | 89.00 |
| Maximum | 1,108,945 | 252.26 | 119.76 |

## Unlabelled visual observations

The deterministic random contact sheet shows Russian and Latin text, numbers,
mixed typography, signs, printed material, low-resolution crops, perspective
distortion, textured and nearly uniform backgrounds, and both 0-degree and
180-degree orientations. These observations are descriptive only; no test
labels were assigned or inferred.

## Preprocessing implications

- Do not resize the data to a square by stretching it.
- Preserve aspect ratio and pad to the model canvas.
- Use a rectangular baseline input; `384x96` and `512x128` are candidates to
  compare during modelling rather than assumptions fixed by the audit.
- Include very long lines and low-height crops in synthetic training data.
- Cover wide brightness, contrast, background, scale, blur, and perspective
  variation in the generator.
- Keep the original archive immutable and copy it from Google Drive to the
  Colab local disk before repeated reads.

The final input size remains an experimental decision and must be selected by
validation Brier score.
