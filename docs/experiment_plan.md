# Experiment plan

Experiments are introduced one controlled change at a time and selected by
validation Brier score.

## Implemented foundation

- Test archive audit and geometry analysis.
- Strict configuration and reproducibility helpers.
- Deterministic synthetic train/validation generation.
- Explicit `x + rot180(x)` paired training and symmetric inference.
- Small-CNN sanity overfit and checkpoint round-trip.
- MobileNetV3-Large and EfficientNet-B0 frozen/fine-tuning pipeline.
- Colab T4 quick-run and artifact export.
- Per-model Drive champions with validation-protocol fingerprints.
- Five-fold OOF comparison of uncalibrated, temperature-scaled, and Platt
  probabilities; the final calibrator is then fitted on all validation data.
- Offline Tesseract baseline comparing confidence at 0° and 180°.
- Observable and resumable training with per-batch progress, atomic
  epoch-boundary recovery, and compatibility checks.
- Unified persistent project layout and self-contained champion bundles.
- Validated registry selection for `best`, named, Drive, and uploaded models.
- Strict ordered access to test images directly inside the ZIP archive.
- Ordered test DataLoader, validated champion loading, symmetric inference,
  calibration, and fingerprinted inference recovery.
- Strict sample-template submission generation and inference diagnostics.
- Qualitative contact sheets and one-click resumable Colab inference for the
  best or a named champion.

Every architecture owns an independent champion. A run can replace only the
champion of the same model and only under a comparable validation protocol.
The checkpoint filename is `<model>_<accuracy>.pt`; promotion is decided by
symmetric Brier score rather than accuracy. `leaderboard.json` summarises all
model champions for later cross-model and ensemble comparison.

## Delivery stages 16–19

16. ~~Add one read-only delivery-readiness command for source, notebooks,
    test data, registry, and champion integrity.~~
17. ~~Cover the reviewer path with an end-to-end test from best-model selection
    through a template-ordered full submission.~~
18. ~~Run the readiness gate and complete test suite in GitHub Actions.~~
19. ~~Document the training/inference hand-off and an explicit final
    pre-submission checklist.~~

These stages make the existing pipeline reviewable; they do not substitute a
quick run for a full experiment or mark unexecuted Optuna/ViT experiments as
complete.

## Remaining experiments

1. Small paired CNN: pipeline sanity only.
2. MobileNetV3-Large baseline with fixed synthetic samples.
3. EfficientNet-B0 under the same split and preprocessing.
4. Best compact CNN with and without symmetry loss.
5. Rectangular input-size comparison.
6. Deterministic train augmentation that changes by epoch while validation
   remains fixed.
7. Direct inference versus symmetric `x + rot180(x)` inference.
8. ~~Probability calibration using validation predictions only.~~ Implemented
   with stratified OOF evaluation to avoid scoring a calibrator on its own fit
   samples.
9. ~~Offline OCR confidence baseline on both orientations.~~ Implemented as a
   separate Tesseract experiment; execution requires the system OCR packages.
10. ~~Observable and resumable Colab training.~~ Implemented with independent
    recovery directories for each architecture.
11. Full EfficientNet-B0 run and compact-model comparison.
12. Optuna search on the best compact architecture. The isolated resumable
    search, pruning, Colab entry point, and full-data confirmation config are
    implemented. Under the deadline budget, only the single best configuration
    is confirmed, and only when the short search improves Brier materially.
13. ViT-B/16 after the compact pipeline is stable. Model construction,
    rectangular positional embeddings, freezing, configuration, training,
    recovery, registry, Colab, and inference are implemented; smoke/full GPU
    experiments remain pending. Local preflight and automated comparable-model
    reporting are implemented for stages 11 and 14.
14. CNN/OCR or CNN/ViT ensemble only if validation Brier improves materially.
15. ~~Final `test.zip` inference and submission creation.~~ Implemented with
    exact template validation, reports, contact sheets, and a reviewer-facing
    Colab notebook; the actual full run remains to be executed after all
    candidate champions are trained.

Epoch-varying augmentation may change brightness, contrast, background,
resolution, blur, noise, JPEG degradation, small angle, and mild perspective.
It must never apply an unlabelled 180-degree rotation or mirror text. Every
epoch remains reproducible through `(global_seed, epoch, sample_id)`.
