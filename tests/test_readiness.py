import json
from pathlib import Path

import pytest

from src.readiness import verify_source


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_source_delivery_is_ready() -> None:
    result = verify_source(PROJECT_ROOT)

    assert result["status"] == "ready"
    assert set(result["configs"]) == {
        "mobilenet_v3_large",
        "efficientnet_b0",
        "vit_b_16",
    }
    assert {Path(item["path"]).name for item in result["notebooks"]} == {
        "colab_train.ipynb",
        "colab_inference.ipynb",
    }


def test_source_delivery_rejects_invalid_notebook(tmp_path: Path) -> None:
    for name in (
        "README.md",
        "requirements.txt",
        "colab_train.ipynb",
        "colab_inference.ipynb",
        "configs/baseline.yaml",
        "configs/efficientnet_b0.yaml",
        "configs/vit_b_16.yaml",
        "scripts/train.py",
        "scripts/infer.py",
    ):
        source = PROJECT_ROOT / name
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    notebook = json.loads((tmp_path / "colab_inference.ipynb").read_text())
    notebook["cells"].append({"cell_type": "code", "source": ["not valid python !"]})
    (tmp_path / "colab_inference.ipynb").write_text(json.dumps(notebook))

    with pytest.raises(SyntaxError):
        verify_source(tmp_path)
