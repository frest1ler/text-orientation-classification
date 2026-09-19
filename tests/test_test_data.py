import csv
import io
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from src.test_data import ZipTestDataset


def image_bytes(mode: str, color) -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, (5, 3), color).save(buffer, format="PNG")
    return buffer.getvalue()


def make_test_zip(path: Path, identifiers: list[str], extra: bool = False) -> None:
    submission = io.StringIO()
    writer = csv.writer(submission)
    writer.writerow(["image_id", "p_180"])
    for identifier in identifiers:
        writer.writerow([identifier, 0.5])
    payloads = [
        image_bytes("RGB", "red"),
        image_bytes("L", 100),
        image_bytes("RGBA", (0, 255, 0, 128)),
    ]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("sample_submission.csv", submission.getvalue())
        for index, identifier in enumerate(identifiers):
            archive.writestr(f"test/images/{identifier}.png", payloads[index % len(payloads)])
        if extra:
            archive.writestr("test/images/extra.png", payloads[0])


def test_dataset_preserves_submission_order_and_converts_rgb(tmp_path: Path) -> None:
    path = tmp_path / "test.zip"
    identifiers = ["third", "first", "second"]
    make_test_zip(path, identifiers)
    dataset = ZipTestDataset(path, "test/images/", "sample_submission.csv")

    assert dataset.image_ids == identifiers
    assert len(dataset) == 3
    for index, identifier in enumerate(identifiers):
        item = dataset[index]
        assert item["image_id"] == identifier
        assert item["image"].mode == "RGB"
        assert item["rotated"].mode == "RGB"
        assert item["image"].size == item["rotated"].size
    dataset.close()


def test_dataset_applies_transform_to_both_orientations(tmp_path: Path) -> None:
    path = tmp_path / "test.zip"
    make_test_zip(path, ["one"])
    dataset = ZipTestDataset(
        path,
        "test/images/",
        "sample_submission.csv",
        transform=lambda image: (image.mode, image.size),
    )
    item = dataset[0]
    assert item["image"] == ("RGB", (5, 3))
    assert item["rotated"] == ("RGB", (5, 3))


def test_dataset_rejects_zip_template_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "test.zip"
    make_test_zip(path, ["one"], extra=True)
    with pytest.raises(ValueError, match="missing=0, extra=1"):
        ZipTestDataset(path, "test/images/", "sample_submission.csv")
