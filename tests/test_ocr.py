from PIL import Image

from src.ocr import orientation_probability, parse_tesseract_confidences


def test_parse_tesseract_confidences_uses_recognised_character_weights() -> None:
    tsv = "level\tconf\ttext\n5\t90\ta\n5\t30\tbbb\n5\t-1\tignored\n"
    assert parse_tesseract_confidences(tsv) == 45.0


def test_parse_tesseract_confidences_handles_no_text() -> None:
    assert parse_tesseract_confidences("level\tconf\ttext\n5\t-1\t\n") == 0.0


def test_orientation_probability_compares_both_orientations() -> None:
    image = Image.new("RGB", (8, 4), "white")
    image.putpixel((0, 0), (0, 0, 0))

    def confidence(candidate: Image.Image) -> float:
        return 80.0 if candidate.getpixel((0, 0)) == (0, 0, 0) else 20.0

    probability, direct, rotated = orientation_probability(image, confidence)
    assert probability < 0.01
    assert direct == 80.0
    assert rotated == 20.0
