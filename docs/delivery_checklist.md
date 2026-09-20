# Delivery checklist

This checklist separates code readiness from expensive model training. Passing
it proves that a reviewer can validate the repository, select a champion, and
produce an ordered submission; it does not claim that every planned candidate
has already completed a full GPU run.

## Automated source check

From the repository root:

```bash
python3 -m scripts.verify_delivery
python3 -m pytest -q
```

The first command validates required entry points, both strict model configs,
the structure of both notebooks, and Python syntax in every notebook code
cell. GitHub Actions runs the same check and the full test suite on every push
and pull request.

## Persistent project check

After mounting Drive or preparing the same layout locally:

```bash
python3 -m scripts.verify_delivery \
  --project-dir /path/to/text-orientation \
  --report /path/to/text-orientation/delivery_report.json
```

This read-only check verifies:

- `data/test.zip`, including exact agreement between its images and submission
  template;
- every leaderboard record and champion manifest;
- checkpoint and calibration SHA-256 hashes;
- selection of the current best champion;
- embedded inference configuration of that champion.

For one uploaded bundle, use the same selection contract as the inference
notebook:

```bash
python3 -m scripts.verify_delivery \
  --project-dir /path/to/text-orientation \
  --artifact-source upload \
  --uploaded-path /path/to/champion-bundle \
  --model mobilenet_v3_large
```

## Reviewer paths

1. Open `notebooks/colab/train.ipynb` to reproduce a quick or full candidate training
   run. Only calibrated full runs may update a per-model champion.
2. Open `notebooks/colab/inference.ipynb`, initially set `SMOKE_IMAGES=64`, and run all
   cells.
3. Set `SMOKE_IMAGES=None` and rerun inference to create the complete
   `submission.csv`.
4. Inspect `inference_report.json` and all four images under
   `contact_sheets/` before using the submission.

The inference notebook accepts either the full Drive registry with
`MODEL="best"`/a named architecture, or one separately uploaded self-contained
champion bundle.

## Final pre-submission gate

- The intended MobileNet, EfficientNet, and any later candidate full runs have
  actually completed; quick runs are not champions.
- The leaderboard winner is chosen by comparable symmetric validation Brier
  score, not test data or visual preference.
- The final inference report records 20,000 images and
  `complete_test_set=true`.
- `submission.csv` has exactly the template header and 20,000 ordered rows.
- No test label was inferred or introduced: the test archive remains
  unlabelled and is used only for prediction and non-training diagnostics.
- The Git commit used in Colab is recorded with the training artifacts.

Optuna, ViT-B/16, and ensembles remain optional experiments. They should be
promoted only after a full run improves the same locked validation protocol.
