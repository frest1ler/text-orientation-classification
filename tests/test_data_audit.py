import io
import zipfile

import pandas as pd
from PIL import Image

from src.data_audit import (
    EXPECTED_COLUMNS,
    SUBMISSION_MEMBER,
    image_id_from_member,
    inspect_image,
    is_safe_member,
    list_image_members,
    read_submission,
    validate_summary,
)


def make_test_archive() -> io.BytesIO:
    buffer = io.BytesIO()
    image_buffer = io.BytesIO()
    Image.new("RGB", (16, 8), "white").save(image_buffer, format="PNG")
    submission = pd.DataFrame({"image_id": ["test_00000"], "p_180": [0.5]})
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(SUBMISSION_MEMBER, submission.to_csv(index=False))
        archive.writestr("test/images/test_00000.png", image_buffer.getvalue())
        archive.writestr("test/images/readme.txt", "not an image")
    buffer.seek(0)
    return buffer


def test_archive_helpers_read_expected_structure() -> None:
    with zipfile.ZipFile(make_test_archive()) as archive:
        members = list_image_members(archive)
        submission = read_submission(archive)

    assert members == ["test/images/test_00000.png"]
    assert submission.columns.tolist() == EXPECTED_COLUMNS
    assert submission["image_id"].tolist() == ["test_00000"]
    assert image_id_from_member(members[0]) == "test_00000"


def test_zip_member_safety() -> None:
    assert is_safe_member("test/images/example.png")
    assert not is_safe_member("../example.png")
    assert not is_safe_member("/absolute/example.png")


def test_image_inspection_preserves_encoded_format() -> None:
    image_buffer = io.BytesIO()
    Image.new("RGB", (16, 8), "white").save(image_buffer, format="PNG")

    result = inspect_image(image_buffer.getvalue(), "test/images/test_00000.png")

    assert result["format"] == "PNG"
    assert result["mode"] == "RGB"
    assert result["width"] == 16
    assert result["height"] == 8


def test_summary_validation_accepts_consistent_dataset() -> None:
    summary = {
        "image_count": 1,
        "decoded_image_count": 1,
        "broken_images": [],
        "submission": {
            "row_count": 1,
            "columns": EXPECTED_COLUMNS,
            "columns_match_expected": True,
            "duplicate_image_ids": 0,
            "missing_images": [],
            "images_not_in_submission": [],
        },
    }

    validate_summary(summary, expected_count=1)
