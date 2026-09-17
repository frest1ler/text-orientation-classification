# Google Colab test run

The first GPU run validates pretrained weight download, CUDA mixed precision,
DataLoader workers, checkpoint persistence, and the real train/validation
path. It does not require `test.zip` because training data is synthetic.

## Before opening Colab

Push the commit containing `colab_train.ipynb` to the public `main` branch.
The notebook clones that branch into a fresh `/content` directory, which
prevents results from depending on a previous Colab session.

## Run

1. Open `colab_train.ipynb` in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Leave the initial parameters as:

   ```python
   MODEL = "mobilenet"
   QUICK_RUN = True
   RUN_TESTS = True
   ```

4. Set `DRIVE_OUTPUT_DIR` if the default Drive folder is unsuitable.
5. Run all cells from top to bottom.

The quick run uses official pretrained weights, 2,048 train pairs, 512
validation pairs, one frozen epoch, and two fine-tuning epochs. It is intended
to expose environment and pipeline failures before spending a full session.

Training is launched as `python -m scripts.train`, so imports resolve from the
repository root without relying on a manually configured `PYTHONPATH`.

## Return artifact

The final cell prints one path similar to:

```text
/content/drive/MyDrive/text-orientation-results/colab_mobilenet_quick.zip
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
