"""Strict, ordered access to test images stored inside the competition ZIP."""

from __future__ import annotations

import csv
import hashlib
import io
import os
import zipfile
from pathlib import Path, PurePosixPath
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset


def zip_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_submission_template(
    archive: zipfile.ZipFile, member: str
) -> tuple[list[str], list[dict[str, str]]]:
    try:
        payload = archive.read(member).decode("utf-8-sig")
    except KeyError as error:
        raise FileNotFoundError(f"Submission template is missing from ZIP: {member}") from error
    reader = csv.DictReader(io.StringIO(payload))
    columns = reader.fieldnames
    if columns is None or len(columns) != 2:
        raise ValueError("sample submission must contain exactly two columns")
    rows = list(reader)
    if not rows:
        raise ValueError("sample submission is empty")
    identifiers = [row[columns[0]] for row in rows]
    if any(not identifier for identifier in identifiers):
        raise ValueError("sample submission contains an empty image identifier")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("sample submission contains duplicate image identifiers")
    return columns, rows


class ZipTestDataset(Dataset):
    """Read RGB test images in exact sample-submission order without extraction."""

    def __init__(
        self,
        zip_path: str | Path,
        image_prefix: str,
        submission_member: str,
        transform: Callable[[Image.Image], object] | None = None,
    ) -> None:
        self.zip_path = Path(zip_path)
        self._archive: zipfile.ZipFile | None = None
        self._archive_pid: int | None = None
        if not self.zip_path.is_file():
            raise FileNotFoundError(f"Test ZIP does not exist: {self.zip_path}")
        if not image_prefix.endswith("/"):
            raise ValueError("image_prefix must end with '/'")
        self.transform = transform
        with zipfile.ZipFile(self.zip_path) as archive:
            self.columns, rows = read_submission_template(archive, submission_member)
            self.template_rows = rows
            members: dict[str, str] = {}
            for name in archive.namelist():
                if name.startswith(image_prefix) and not name.endswith("/"):
                    identifier = PurePosixPath(name).stem
                    if identifier in members:
                        raise ValueError(f"Duplicate image identifier in ZIP: {identifier}")
                    members[identifier] = name
        self.image_ids = [row[self.columns[0]] for row in self.template_rows]
        expected = set(self.image_ids)
        missing = sorted(expected - set(members))
        extra = sorted(set(members) - expected)
        if missing or extra:
            raise ValueError(
                f"ZIP/template image mismatch: missing={len(missing)}, extra={len(extra)}"
            )
        self.members = [members[identifier] for identifier in self.image_ids]

    def __len__(self) -> int:
        return len(self.image_ids)

    def _get_archive(self) -> zipfile.ZipFile:
        pid = os.getpid()
        if self._archive is None or self._archive_pid != pid:
            if self._archive is not None:
                self._archive.close()
            self._archive = zipfile.ZipFile(self.zip_path)
            self._archive_pid = pid
        return self._archive

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        payload = self._get_archive().read(self.members[index])
        try:
            with Image.open(io.BytesIO(payload)) as decoded:
                image = decoded.convert("RGB")
        except Exception as error:
            raise ValueError(f"Failed to decode test image: {self.members[index]}") from error
        rotated = image.transpose(Image.Transpose.ROTATE_180)
        if self.transform is not None:
            image = self.transform(image)
            rotated = self.transform(rotated)
        return {
            "image_id": self.image_ids[index],
            "image": image,
            "rotated": rotated,
        }

    def close(self) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None
            self._archive_pid = None

    def __del__(self) -> None:
        self.close()
