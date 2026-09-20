# Persistent project layout

Training and inference use one portable root directory. In Colab the default
is `/content/drive/MyDrive/text-orientation`; a reviewer may point
`PROJECT_DIR` at any other mounted or local directory.

```text
text-orientation/
  data/
    test.zip
  registry/
    leaderboard.json
    champions/
      <model>/
        <model>_<accuracy>.pt
        calibration.json
        champion.json
    robust/
      leaderboard.json
      champions/
        mobilenet_v3_large/
  training/
    runs/
      <model>/<quick|full>/
      robust/mobilenet_v3_large/<quick|full>/
    recovery/
      <model>/<quick|full>/
      robust/mobilenet_v3_large/<quick|full>/
  inference/
    runs/
      <run-id>/
        submission.csv
        predictions.csv
        inference_report.json
        contact_sheets/
    recovery/
      <model>/<full|smoke_N>/
  evaluation/
    champions/
      champion_comparison.json
      champion_comparison.csv
      champion_comparison.md
  ocr/
    runs/
```

The model registry is portable on its own. A client can use the entire
`registry/` directory and select `best` or a named architecture, or use one
uploaded `champions/<model>/` bundle. Selection verifies the leaderboard,
checkpoint hash, calibration hash, checkpoint epoch, and calibration mode.

## One-time transition from the legacy layout

Migration is deliberately non-destructive. Copy `test.zip` to `data/`, then
rebuild the MobileNet champion bundle from the previously saved full run ZIP:

```bash
python3 -m scripts.import_champion \
  --run /path/to/colab_mobilenet_full.zip \
  --project-dir /path/to/text-orientation
```

This creates `registry/leaderboard.json` and a self-contained MobileNet bundle
including the calibration artifact. The source ZIP and all legacy Drive files
remain untouched. Old files should be removed only after the new bundle passes
hash validation and the full run archive exists in `training/runs/`.

Future full training runs write directly to the unified structure and require
no migration.
