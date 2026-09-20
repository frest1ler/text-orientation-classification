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
    assert {
        str(Path(item["path"]).relative_to(PROJECT_ROOT))
        for item in result["notebooks"]
    } == {
        "notebooks/colab/train.ipynb",
        "notebooks/colab/optuna.ipynb",
        "notebooks/colab/robust_train.ipynb",
        "notebooks/colab/inference.ipynb",
        "notebooks/colab/ensemble.ipynb",
        "notebooks/cloudcompute/train.ipynb",
        "notebooks/cloudcompute/optuna.ipynb",
        "notebooks/cloudcompute/robust_train.ipynb",
        "notebooks/cloudcompute/inference.ipynb",
        "notebooks/cloudcompute/ensemble.ipynb",
    }
    train_notebook = (PROJECT_ROOT / "notebooks/colab/train.ipynb").read_text()
    inference_notebook = (PROJECT_ROOT / "notebooks/colab/inference.ipynb").read_text()
    robust_notebook = (PROJECT_ROOT / "notebooks/colab/robust_train.ipynb").read_text()
    assert "configs/vit_b_16.yaml" in train_notebook
    assert "TRAIN_BATCH_SIZE = None" in train_notebook
    assert "vit_b_16" in inference_notebook
    assert "BATCH_SIZE = None" in inference_notebook
    assert 'MODEL = \\"vit\\"  # vit | mobilenet' in robust_notebook
    assert "configs/vit_b_16.yaml" in robust_notebook
    assert "vit_b_16_robust" in inference_notebook


def test_source_delivery_rejects_invalid_notebook(tmp_path: Path) -> None:
    for name in (
        "README.md",
        "requirements.txt",
        "notebooks/colab/train.ipynb",
        "notebooks/colab/optuna.ipynb",
        "notebooks/colab/robust_train.ipynb",
        "notebooks/colab/inference.ipynb",
        "notebooks/colab/ensemble.ipynb",
        "notebooks/cloudcompute/train.ipynb",
        "notebooks/cloudcompute/optuna.ipynb",
        "notebooks/cloudcompute/robust_train.ipynb",
        "notebooks/cloudcompute/inference.ipynb",
        "notebooks/cloudcompute/ensemble.ipynb",
        "configs/baseline.yaml",
        "configs/efficientnet_b0.yaml",
        "configs/vit_b_16.yaml",
        "scripts/train.py",
        "scripts/optuna_search.py",
        "scripts/train_robust.py",
        "scripts/promote_robust_champion.py",
        "scripts/compare_robust.py",
        "scripts/infer.py",
        "scripts/infer_ensemble.py",
        "scripts/search_ensemble.py",
        "scripts/compare_champions.py",
    ):
        source = PROJECT_ROOT / name
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    notebook_path = tmp_path / "notebooks/colab/inference.ipynb"
    notebook = json.loads(notebook_path.read_text())
    notebook["cells"].append({"cell_type": "code", "source": ["not valid python !"]})
    notebook_path.write_text(json.dumps(notebook))

    with pytest.raises(SyntaxError):
        verify_source(tmp_path)
