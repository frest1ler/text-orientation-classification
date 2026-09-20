# Probability ensembles

Ensembles are isolated from individual and robust champion registries. They
never overwrite a model checkpoint. A promoted ensemble is a small manifest
under `registry/ensembles/` that pins component checkpoint/calibration hashes,
positive weights, the validation protocol, and validation metrics.

## Weight search

Extract each completed run so that `best.pt`, `calibration.json`, and
`validation_predictions.npz` are in one directory. Then run:

```bash
python3 -m scripts.search_ensemble \
  --project-dir /path/to/text-orientation \
  --candidate mobilenet_v3_large_robust=/path/to/robust-run \
  --candidate vit_b_16=/path/to/vit-run \
  --weight-step 0.05
```

The search uses leakage-resistant out-of-fold calibrated probabilities. It
rejects different validation fingerprints, target arrays, or sample order.
Legacy prediction archives without `sample_ids` use deterministic validation
row indices and still require identical targets and protocol hashes.

For two models use a `0.05` step. For three models use `0.10` to keep the
search intentionally small. Only strictly positive weights are considered.

## Test inference

Choose `MODEL="best_ensemble"` in the normal inference notebook, first with
`SMOKE_IMAGES=64`, then with `SMOKE_IMAGES=None`. Components are executed
sequentially through the existing resumable inference pipeline. Their ordered
probabilities are cached independently and then mixed, so an interruption does
not discard completed component work.
