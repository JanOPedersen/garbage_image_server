import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app import create_app
from app.config import Config
from app.services.model_registry import model_registry


class SmokeConfig(Config):
    TESTING = True


def _make_test_image_bytes(width: int = 512, height: int = 512) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 1] = 80
    image[..., 2] = 40

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_predict_smoke_with_real_onnx(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = repo_root / "models" / "manifests" / "cigarette-butt-v1.yaml"
    artifacts_dir = repo_root / "app" / "model_artifacts"
    onnx_path = artifacts_dir / "cigarette-butt" / "cigbutts_yolo11n.onnx"

    if not onnx_path.exists():
        pytest.skip(f"Smoke test skipped; ONNX artifact not found: {onnx_path}")

    monkeypatch.setenv("MODEL_ARTIFACTS_DIR", str(artifacts_dir))

    # Ensure clean singleton state for this test.
    model_registry._models = {}
    model_registry._default_model_id = None

    model_registry.load_from_manifest_file(str(manifest_path))

    app = create_app(SmokeConfig)
    client = app.test_client()

    image_bytes = _make_test_image_bytes()
    response = client.open(
        "/predict",
        method="GET",
        data={"image": (io.BytesIO(image_bytes), "smoke.jpg")},
        content_type="multipart/form-data",
    )

    payload = response.get_json()

    assert response.status_code == 200
    assert payload["model_id"] == "cigarette-butt-v1"
    assert "request_id" in payload
    assert "timing_ms" in payload
    assert payload["image"]["filename"] == "smoke.jpg"
    assert payload["image"]["width"] == 512
    assert payload["image"]["height"] == 512
    assert isinstance(payload["detections"], list)

    # Avoid leaking loaded singleton state across tests.
    model_registry._models = {}
    model_registry._default_model_id = None
