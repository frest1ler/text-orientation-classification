# Google Colab test run

The first GPU run validates pretrained weight download, CUDA mixed precision,
DataLoader workers, checkpoint persistence, and the real train/validation
path. It does not require `test.zip` because training data is synthetic.

## Before opening Colab

Push the commit containing `notebooks/colab/train.ipynb` to the public `main` branch.
The notebook clones that branch into a fresh `/content` directory, which
prevents results from depending on a previous Colab session.

## Run

1. Open `notebooks/colab/train.ipynb` in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Leave the initial parameters as:

   ```python
   MODEL = "mobilenet"
   QUICK_RUN = True
   PROMOTE_CHAMPION = not QUICK_RUN
   RUN_CALIBRATION = not QUICK_RUN
   RESUME_TRAINING = True
   RUN_TESTS = True
   PROJECT_DIR = "/content/drive/MyDrive/text-orientation"
   ```

4. Set `PROJECT_DIR` if the default Drive folder is unsuitable. All persistent
   data and results live below this one directory.
5. Run all cells from top to bottom.

The quick run uses official pretrained weights, 2,048 train pairs, 512
validation pairs, one frozen epoch, and two fine-tuning epochs. It is intended
to expose environment and pipeline failures before spending a full session.
For ViT-B/16, set `MODEL="vit"`; its quick batches are 16/32 instead of the
CNN 64/128. If T4 memory is insufficient, start a new run with explicit 8/16
batches rather than silently changing an existing recovery trajectory.

Training is launched as `python -m scripts.train`, so imports resolve from the
repository root without relying on a manually configured `PYTHONPATH`.

## Return artifact

The final cell prints one path similar to:

```text
/content/drive/MyDrive/text-orientation/training/runs/mobilenet_v3_large/quick/colab_mobilenet_quick.zip
```

Send that ZIP back unchanged. It contains:

- `best.pt` — best checkpoint by symmetric Brier score;
- `history.json` — per-epoch direct and symmetric metrics;
- `config.json` — committed experiment configuration;
- `runtime.json` — actual quick-run overrides and device;
- `environment.json` — Python, PyTorch, CUDA, GPU, and Git commit;
- `colab.log` — complete combined stdout/stderr.

No token, Google credentials, test archive, or whole Drive folder should be
sent. If the notebook fails before creating the ZIP, send the complete error
trace and the output of the GPU/test cell.

After inspecting this artifact, the next decision is whether to run the full
MobileNet experiment or first correct the Colab environment/pipeline. The
EfficientNet run remains disabled until MobileNet succeeds.

## Per-model champions

Quick runs are archived but never promoted. With `QUICK_RUN=False`, the
default `PROMOTE_CHAMPION=True` compares the completed run with the champion
of the same architecture. Promotion uses lower symmetric Brier score, then
lower log loss, higher ROC-AUC, and higher accuracy as tie-breakers.

```text
text-orientation/
  data/
    test.zip
  registry/
    leaderboard.json
    champions/
      mobilenet_v3_large/
        mobilenet_v3_large_<accuracy>.pt
        calibration.json
        champion.json
      efficientnet_b0/
        efficientnet_b0_<accuracy>.pt
        calibration.json
        champion.json
      vit_b_16/
        vit_b_16_<accuracy>.pt
        calibration.json
        champion.json
  training/
    runs/
    recovery/
  inference/
    runs/
    recovery/
  ocr/
    runs/
```

Promotion is rejected when validation seed, sample count, synthetic settings,
generator source, corpus source, or font manifest differ from the existing
champion. The run ZIP remains available even when it is not promoted.

## Calibration artifact

Full runs execute five-fold stratified OOF calibration before creating the
ZIP. `calibration.json` compares raw, temperature-scaled, and Platt-scaled
probabilities. `validation_predictions.npz` contains the corresponding OOF
predictions. The selected calibrator is fitted on all validation predictions
for later test inference, but calibrated metrics do not replace the raw
champion score.

The OCR baseline is intentionally not part of every training run. In a
separate Colab cell install and run it with:

```python
!apt-get -qq update
!apt-get -qq install -y tesseract-ocr tesseract-ocr-rus
!python -m scripts.ocr_baseline --config configs/baseline.yaml --base-samples 500
```

## Progress and recovery

Train and validation display a batch progress bar with percentage, speed, ETA,
and running losses. After every completed epoch, the notebook atomically
updates the architecture-specific Drive directory:

```text
text-orientation/
  training/
    recovery/
      mobilenet_v3_large/
        quick/
          last.pt
          best.pt
          status.json
        full/
          last.pt
          best.pt
          status.json
      efficientnet_b0/
        quick/
        full/
      vit_b_16/
        quick/
        full/
```

With `RESUME_TRAINING=True`, a new Colab session resumes at the next epoch.
The loader generator and PyTorch CPU/CUDA RNG states are restored together
with the optimizer and early-stopping state. A completed recovery is reused
without repeating training. The checkpoint remains on Drive after success.

Resume intentionally fails when model settings, sample counts, batch sizes,
epoch budgets, or critical training source files changed. To deliberately
start a different experiment, choose a new recovery directory or manually
remove the old architecture recovery after preserving its run archive.

Quick and full runs use separate recovery directories, so a completed smoke
test can never block or be mistaken for a full training checkpoint. Legacy
checkpoints directly under the legacy `recovery/<model>/` are not migrated
automatically because their run mode cannot be inferred safely.

Exact bitwise equality is expected only in the same software/hardware runtime;
across different GPU types the trajectory remains controlled but floating
point kernels may still produce small numerical differences.
