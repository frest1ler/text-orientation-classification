"""Audit the competition test data directly inside its ZIP archive."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps


SUBMISSION_MEMBER = "sample_submission.csv"
IMAGE_PREFIX = "test/images/"
EXPECTED_COLUMNS = ["image_id", "p_180"]
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def is_safe_member(name: str) -> bool:
    """Return whether a ZIP member is a safe relative POSIX path."""
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def list_image_members(archive: zipfile.ZipFile) -> list[str]:
    """Return sorted image member names from the expected directory."""
    return sorted(
        info.filename
        for info in archive.infolist()
        if not info.is_dir()
        and info.filename.startswith(IMAGE_PREFIX)
        and PurePosixPath(info.filename).suffix.lower() in SUPPORTED_SUFFIXES
    )


def read_submission(archive: zipfile.ZipFile) -> pd.DataFrame:
    """Read the sample submission without extracting it."""
    with archive.open(SUBMISSION_MEMBER) as stream:
        return pd.read_csv(stream)


def image_id_from_member(member: str) -> str:
    """Convert an archive member path to its submission image ID."""
    return PurePosixPath(member).stem


def difference_hash(image: Image.Image, hash_size: int = 8) -> str:
    """Compute a small perceptual dHash for duplicate candidate discovery."""
    gray = ImageOps.grayscale(image).resize(
        (hash_size + 1, hash_size), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(gray)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = sum(int(bit) << index for index, bit in enumerate(bits.ravel()))
    return f"{value:0{hash_size * hash_size // 4}x}"


def inspect_image(raw: bytes, member: str) -> dict[str, Any]:
    """Decode one image and return compact metadata and image statistics."""
    sha256 = hashlib.sha256(raw).hexdigest()
    with Image.open(io.BytesIO(raw)) as opened:
        image_format = opened.format
        opened.verify()
    with Image.open(io.BytesIO(raw)) as opened:
        image = opened.copy()
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
        width, height = image.size
        return {
            "member": member,
            "image_id": image_id_from_member(member),
            "format": image_format,
            "mode": image.mode,
            "width": width,
            "height": height,
            "aspect_ratio": width / height,
            "file_size": len(raw),
            "pixel_mean": float(array.mean()),
            "pixel_std": float(array.std()),
            "sha256": sha256,
            "dhash": difference_hash(image),
        }


def duplicate_groups(values: Iterable[str], image_ids: Iterable[str]) -> list[list[str]]:
    """Group IDs sharing the same hash, omitting unique hashes."""
    groups: dict[str, list[str]] = defaultdict(list)
    for value, image_id in zip(values, image_ids, strict=True):
        groups[value].append(image_id)
    return [ids for ids in groups.values() if len(ids) > 1]


def numeric_summary(series: pd.Series) -> dict[str, float]:
    """Return JSON-friendly distribution statistics."""
    quantiles = series.quantile([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    return {
        "min": float(quantiles.loc[0.0]),
        "p01": float(quantiles.loc[0.01]),
        "p05": float(quantiles.loc[0.05]),
        "p25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.5]),
        "p75": float(quantiles.loc[0.75]),
        "p95": float(quantiles.loc[0.95]),
        "p99": float(quantiles.loc[0.99]),
        "max": float(quantiles.loc[1.0]),
        "mean": float(series.mean()),
    }


def make_contact_sheet(
    archive: zipfile.ZipFile,
    members: list[str],
    output_path: Path,
    seed: int,
    sample_size: int = 24,
) -> list[str]:
    """Save a deterministic, unlabelled random sample as a contact sheet."""
    rng = random.Random(seed)
    chosen = rng.sample(members, k=min(sample_size, len(members)))
    tile_size = (320, 120)
    columns = 4
    rows = (len(chosen) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_size[0], rows * tile_size[1]), "#dddddd")
    draw = ImageDraw.Draw(sheet)

    for index, member in enumerate(chosen):
        with archive.open(member) as stream, Image.open(stream) as opened:
            image = opened.convert("RGB")
            image.thumbnail((tile_size[0] - 12, tile_size[1] - 28), Image.Resampling.LANCZOS)
        tile_x = (index % columns) * tile_size[0]
        tile_y = (index // columns) * tile_size[1]
        x = tile_x + (tile_size[0] - image.width) // 2
        y = tile_y + 4 + (tile_size[1] - 24 - image.height) // 2
        sheet.paste(image, (x, y))
        draw.text((tile_x + 6, tile_y + tile_size[1] - 18), image_id_from_member(member), fill="black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, optimize=True)
    return [image_id_from_member(member) for member in chosen]


def audit_archive(zip_path: Path, output_dir: Path, seed: int = 42) -> dict[str, Any]:
    """Run the complete archive audit and persist compact reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        unsafe = [name for name in names if not is_safe_member(name)]
        duplicate_members = [name for name, count in Counter(names).items() if count > 1]
        if unsafe:
            raise ValueError(f"Unsafe archive member paths: {unsafe[:5]}")
        if duplicate_members:
            raise ValueError(f"Duplicate archive member names: {duplicate_members[:5]}")
        if SUBMISSION_MEMBER not in names:
            raise FileNotFoundError(f"Missing {SUBMISSION_MEMBER} in {zip_path}")

        submission = read_submission(archive)
        members = list_image_members(archive)
        rows: list[dict[str, Any]] = []
        broken: list[dict[str, str]] = []
        for member in members:
            try:
                rows.append(inspect_image(archive.read(member), member))
            except Exception as error:  # collect all failures for one useful report
                broken.append({"member": member, "error": repr(error)})

        metadata = pd.DataFrame(rows)
        image_ids = metadata["image_id"].tolist()
        submission_ids = submission["image_id"].astype(str).tolist() if "image_id" in submission else []
        image_id_set = set(image_ids)
        submission_id_set = set(submission_ids)
        exact_groups = duplicate_groups(metadata["sha256"], metadata["image_id"])
        perceptual_groups = duplicate_groups(metadata["dhash"], metadata["image_id"])
        sample_ids = make_contact_sheet(
            archive, members, output_dir / "random_sample.png", seed=seed
        )

    summary = {
        "archive": str(zip_path),
        "archive_size_bytes": zip_path.stat().st_size,
        "archive_member_count": len(names),
        "image_count": len(members),
        "decoded_image_count": len(metadata),
        "broken_images": broken,
        "submission": {
            "columns": submission.columns.tolist(),
            "row_count": len(submission),
            "unique_image_ids": int(submission["image_id"].nunique()) if "image_id" in submission else 0,
            "duplicate_image_ids": int(submission["image_id"].duplicated().sum()) if "image_id" in submission else 0,
            "missing_images": sorted(submission_id_set - image_id_set),
            "images_not_in_submission": sorted(image_id_set - submission_id_set),
            "order_matches_archive_sort": submission_ids == image_ids,
            "columns_match_expected": submission.columns.tolist() == EXPECTED_COLUMNS,
        },
        "formats": metadata["format"].value_counts().to_dict(),
        "modes": metadata["mode"].value_counts().to_dict(),
        "width": numeric_summary(metadata["width"]),
        "height": numeric_summary(metadata["height"]),
        "aspect_ratio": numeric_summary(metadata["aspect_ratio"]),
        "file_size": numeric_summary(metadata["file_size"]),
        "pixel_mean": numeric_summary(metadata["pixel_mean"]),
        "pixel_std": numeric_summary(metadata["pixel_std"]),
        "exact_duplicate_group_count": len(exact_groups),
        "exact_duplicate_image_count": sum(map(len, exact_groups)),
        "perceptual_hash_collision_group_count": len(perceptual_groups),
        "perceptual_hash_collision_image_count": sum(map(len, perceptual_groups)),
        "exact_duplicate_groups": exact_groups,
        "perceptual_hash_collision_groups": perceptual_groups,
        "random_sample_seed": seed,
        "random_sample_image_ids": sample_ids,
    }

    metadata.to_csv(output_dir / "image_metadata.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    return summary


def validate_summary(summary: dict[str, Any], expected_count: int) -> None:
    """Fail clearly when the competition data violates core invariants."""
    submission = summary["submission"]
    errors = []
    if summary["image_count"] != expected_count:
        errors.append(f"expected {expected_count} images, found {summary['image_count']}")
    if summary["decoded_image_count"] != summary["image_count"]:
        errors.append(f"failed to decode {len(summary['broken_images'])} images")
    if submission["row_count"] != expected_count:
        errors.append(f"expected {expected_count} submission rows, found {submission['row_count']}")
    if not submission["columns_match_expected"]:
        errors.append(f"unexpected submission columns: {submission['columns']}")
    if submission["duplicate_image_ids"]:
        errors.append("sample submission contains duplicate image IDs")
    if submission["missing_images"] or submission["images_not_in_submission"]:
        errors.append("sample submission and archive image IDs differ")
    if errors:
        raise ValueError("; ".join(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", type=Path, default=Path("test.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/data_audit"))
    parser.add_argument("--expected-count", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = audit_archive(args.zip_path, args.output_dir, seed=args.seed)
    validate_summary(summary, expected_count=args.expected_count)
    print(json.dumps({
        "image_count": summary["image_count"],
        "broken_image_count": len(summary["broken_images"]),
        "exact_duplicate_group_count": summary["exact_duplicate_group_count"],
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
