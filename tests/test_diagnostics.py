import csv
import io
import zipfile

from PIL import Image

from src.diagnostics import build_inference_report, create_contact_sheets
from src.test_data import ZipTestDataset


def _make_zip(path) -> None:
    submission = io.StringIO()
    writer = csv.writer(submission)
    writer.writerow(["image_id", "p_180"])
    payload = io.BytesIO()
    Image.new("RGB", (30, 10), "white").save(payload, format="PNG")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("sample_submission.csv", submission.getvalue() + "a,0.5\nb,0.5\n")
        archive.writestr("test/images/a.png", payload.getvalue())
        archive.writestr("test/images/b.png", payload.getvalue())


def _rows():
    return [
        {"image_id": "a", "p_symmetric": 0.1, "p_final": 0.2, "symmetry_error": 0.05},
        {"image_id": "b", "p_symmetric": 0.9, "p_final": 0.8, "symmetry_error": 0.4},
    ]


def test_report_contains_provenance_timing_and_distributions() -> None:
    report = build_inference_report(
        _rows(),
        {"test_zip_sha256": "test-hash", "elapsed_seconds": 2.0},
        {
            "model": "small_cnn",
            "checkpoint": "checkpoint.pt",
            "checkpoint_sha256": "checkpoint-hash",
            "metrics": {"symmetric": {"brier_score": 0.1}},
        },
        {"final_calibrator": {"method": "temperature", "slope": 0.5, "intercept": 0.0}},
    )

    assert report["images"] == 2
    assert report["images_per_second"] == 1.0
    assert report["final_probability"]["median"] == 0.5
    assert report["fractions"]["symmetry_error_above_0.25"] == 0.5


def test_contact_sheets_are_created(tmp_path) -> None:
    archive = tmp_path / "test.zip"
    _make_zip(archive)
    dataset = ZipTestDataset(archive, "test/images/", "sample_submission.csv")

    paths = create_contact_sheets(dataset, _rows(), tmp_path / "sheets", count=2)

    assert len(paths) == 4
    assert all(path.is_file() for path in paths)
    assert all(Image.open(path).format == "PNG" for path in paths)
    dataset.close()
