# Resumable inference and submission

The inference stage produces a competition-ready submission and a reproducible
diagnostic package. `colab_inference.ipynb` is the one-click reviewer entry
point.

## Inputs

- unified `PROJECT_DIR`;
- either its `registry/` directory or one uploaded champion bundle;
- `data/test.zip` or an explicit test ZIP path;
- `best` or a concrete architecture name.

The loader verifies checkpoint and calibration hashes, opens the checkpoint
without downloading pretrained weights, checks its epoch, metrics, model name,
and embedded configuration, then recreates the training preprocessing.
Unless `--batch-size` is supplied, inference uses the batch size persisted in
the selected champion configuration. This matters when `best` resolves to the
heavier ViT-B/16 candidate.

## Symmetric prediction

For every test image the pipeline stores direct, rotated, symmetric, symmetry
error, and calibrated probabilities. Calibration is applied only after the
symmetric probability is formed.

```text
p_symmetric = 0.5 * (p_direct + 1 - p_rotated)
p_final = calibrator(p_symmetric)
```

## Recovery

`inference/recovery/<model>/<full|smoke_N>/` contains atomic `predictions.csv` and
`metadata.json`. The committed row count in metadata is authoritative; a CSV
suffix written immediately before an interruption is discarded safely.
Resume verifies test ZIP, checkpoint, calibration, preprocessing, batch size,
worker count, requested limit, and critical inference source hashes.

Separating full and smoke recovery prevents a short verification run from
blocking or being mistaken for a full run.

## Final artifacts

A complete run stores the following below `inference/runs/<run-id>/`:

- `submission.csv`, preserving the exact sample-submission columns, row count,
  image identifiers, and order;
- `predictions.csv` with direct, rotated, symmetric, calibrated probabilities,
  and symmetry error;
- `inference_report.json` with bundle/test hashes, validation metrics,
  calibrator, timing, probability quantiles, and uncertainty fractions;
- `contact_sheets/` with the most confidently upright, confidently upside
  down, uncertain, and symmetry-violating examples.

`--limit N` is explicitly a smoke run. It produces predictions, a report, and
contact sheets, but never a misleading partial `submission.csv`.

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

For a smoke check, add `--limit 64`; it automatically uses a separate
`smoke_64` recovery directory. Use `--run-name NAME` only when a stable custom
output directory is useful.

## Colab reviewer flow

Open `colab_inference.ipynb`, set `PROJECT_DIR`, and choose:

- `MODEL="best"` for the registry winner or a concrete architecture name;
- `ARTIFACT_SOURCE="drive"` for `PROJECT_DIR/registry`;
- `ARTIFACT_SOURCE="upload"` plus `UPLOADED_PATH` for one self-contained
  champion directory.

Keep `SMOKE_IMAGES=64` for the first check, then set it to `None` for the full
submission. Recovery and final outputs remain on Google Drive.
