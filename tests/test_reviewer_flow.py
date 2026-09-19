import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import torch
from PIL import Image

from scripts.infer import main as infer_main
from src.models import build_model
from src.readiness import verify_project
from src.registry import file_sha256


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_test_zip(path: Path) -> None:
    submission = io.StringIO()
    writer = csv.writer(submission)
    writer.writerow(["image_id", "p_180"])
    writer.writerows((("second", 0.5), ("first", 0.5)))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("sample_submission.csv", submission.getvalue())
        for image_id, color in (("first", "white"), ("second", "black")):
            payload = io.BytesIO()
            Image.new("RGB", (48, 16), color).save(payload, format="PNG")
            archive.writestr(f"test/images/{image_id}.png", payload.getvalue())


def _make_project(root: Path) -> None:
    data = root / "data"
    bundle = root / "registry" / "champions" / "small_cnn"
    data.mkdir(parents=True)
    bundle.mkdir(parents=True)
    _make_test_zip(data / "test.zip")
    model = build_model("small_cnn", dropout=0.0, pretrained=False)
    metrics = {
        "symmetric": {
            "brier_score": 0.1,
            "log_loss": 0.3,
            "roc_auc": 0.9,
            "accuracy": 0.8,
        }
    }
    config = {
        "model": {
            "name": "small_cnn",
            "dropout": 0.0,
            "input_height": 32,
            "input_width": 64,
        },
        "data": {
            "image_prefix": "test/images/",
            "sample_submission_member": "sample_submission.csv",
        },
    }
    checkpoint = bundle / "small_cnn_0.800000.pt"
    torch.save(
        {"model_state": model.state_dict(), "epoch": 1, "metrics": metrics, "config": config},
        checkpoint,
    )
    calibration = bundle / "calibration.json"
    _write_json(
        calibration,
        {
            "checkpoint_epoch": 1,
            "prediction_mode": "symmetric",
            "final_calibrator": {
                "method": "uncalibrated",
                "slope": 1.0,
                "intercept": 0.0,
            },
        },
    )
    manifest = {
        "model": "small_cnn",
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": file_sha256(checkpoint),
        "calibration": calibration.name,
        "calibration_sha256": file_sha256(calibration),
        "epoch": 1,
        "metrics": metrics,
        "validation_protocol_sha256": "test-protocol",
    }
    _write_json(bundle / "champion.json", manifest)
    record = {key: manifest[key] for key in manifest if key != "model"}
    _write_json(root / "registry" / "leaderboard.json", {"models": {"small_cnn": record}})


def test_reviewer_can_verify_and_run_full_inference(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    _make_project(project)

    readiness = verify_project(project)
    assert readiness["selected_model"] == "small_cnn"
    assert readiness["test_images"] == 2
    uploaded = verify_project(
        project,
        artifact_source="upload",
        uploaded_path=project / "registry" / "champions" / "small_cnn",
        model="small_cnn",
    )
    assert uploaded["artifact_source"] == "upload"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts.infer",
            "--project-dir",
            str(project),
            "--model",
            "best",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--skip-contact-sheets",
        ],
    )
    infer_main()

    run = next((project / "inference" / "runs").iterdir())
    with (run / "submission.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["image_id"] for row in rows] == ["second", "first"]
    assert (run / "predictions.csv").is_file()
    assert json.loads((run / "inference_report.json").read_text())["complete_test_set"]
