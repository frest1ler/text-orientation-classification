# Resumable inference foundation

The current inference stage produces validated technical predictions. Final
submission creation, diagnostics, contact sheets, and the one-click Colab
notebook are implemented in the following stages.

## Inputs

- unified `PROJECT_DIR`;
- either its `registry/` directory or one uploaded champion bundle;
- `data/test.zip` or an explicit test ZIP path;
- `best` or a concrete architecture name.

The loader verifies checkpoint and calibration hashes, opens the checkpoint
without downloading pretrained weights, checks its epoch, metrics, model name,
and embedded configuration, then recreates the training preprocessing.

## Symmetric prediction

For every test image the pipeline stores direct, rotated, symmetric, symmetry
error, and calibrated probabilities. Calibration is applied only after the
symmetric probability is formed.

```text
p_symmetric = 0.5 * (p_direct + 1 - p_rotated)
p_final = calibrator(p_symmetric)
```

## Recovery

`inference/recovery/<model>/` contains atomic `predictions.csv` and
`metadata.json`. The committed row count in metadata is authoritative; a CSV
suffix written immediately before an interruption is discarded safely.
Resume verifies test ZIP, checkpoint, calibration, preprocessing, batch size,
worker count, requested limit, and critical inference source hashes.

## CLI

Use the unified registry:

```bash
python3 -m scripts.infer \
  --project-dir /path/to/text-orientation \
  --artifact-source drive \
  --model best
```

Use one manually provided bundle:

```bash
python3 -m scripts.infer \
  --project-dir /path/to/text-orientation \
  --artifact-source upload \
  --uploaded-path /path/to/champion-bundle \
  --model mobilenet_v3_large
```

For a smoke check, add `--limit 64`. A changed limit intentionally requires a
separate recovery directory or removal of the incompatible smoke recovery.
