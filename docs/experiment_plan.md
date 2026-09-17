# Experiment plan

Experiments are introduced one controlled change at a time and selected by
validation Brier score.

1. Small paired CNN: pipeline sanity only.
2. MobileNetV3-Large baseline with fixed synthetic samples.
3. EfficientNet-B0 under the same split and preprocessing.
4. Best compact CNN with and without symmetry loss.
5. Rectangular input-size comparison.
6. Deterministic train augmentation that changes by epoch while validation
   remains fixed.
7. Direct inference versus symmetric `x + rot180(x)` inference.
8. Probability calibration using validation predictions only.
9. Offline OCR confidence baseline on both orientations.
10. ViT-B/16 after the compact pipeline is stable.
11. CNN/OCR or CNN/ViT ensemble only if validation Brier improves materially.

Epoch-varying augmentation may change brightness, contrast, background,
resolution, blur, noise, JPEG degradation, small angle, and mild perspective.
It must never apply an unlabelled 180-degree rotation or mirror text. Every
epoch remains reproducible through `(global_seed, epoch, sample_id)`.
