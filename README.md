# text-orientation-classification
Avito Bootcamp test task: text orientation classification.

## Data audit

The test archive is treated as immutable and read directly with Python's
`zipfile` module. Run the reproducible integrity and metadata audit with:

```bash
PYTHONPATH=. python3 -m src.data_audit \
  --zip-path test.zip \
  --output-dir artifacts/data_audit
```

The archive itself, per-image metadata, and contact sheet are ignored by Git.
The compact aggregate report is stored in
`artifacts/data_audit/summary.json`. See [docs/data_audit.md](docs/data_audit.md)
for findings from the initial audit.

Run the current tests with:

```bash
PYTHONPATH=. python3 -m pytest -q
```
