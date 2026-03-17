from pathlib import Path

import yaml

from app.services import model_registry as model_registry_module
from app.services.model_registry import ModelRegistry


class DummyONNXRuntimeService:
    def __init__(self, manifest):
        self.manifest = manifest
        self.loaded = False

    def load(self):
        self.loaded = True

    def warmup(self):
        return None

    def metadata(self):
        return {
            "model_id": self.manifest["model_id"],
            "loaded": self.loaded,
        }


def test_model_registry_loads_only_configured_manifest(tmp_path, monkeypatch):
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()

    selected_manifest = manifest_dir / "selected.yaml"
    other_manifest = manifest_dir / "other.yaml"

    selected_manifest.write_text(
        yaml.safe_dump({"model_id": "selected-model"}),
        encoding="utf-8",
    )
    other_manifest.write_text(
        yaml.safe_dump({"model_id": "other-model"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        model_registry_module,
        "ONNXRuntimeService",
        DummyONNXRuntimeService,
    )

    registry = ModelRegistry()
    registry.load_from_manifest_file(str(selected_manifest))

    assert registry.default_model_id == "selected-model"
    assert registry.list_models() == ["selected-model"]
    assert registry.get("selected-model") is not None
    assert registry.get("other-model") is None


def test_model_registry_raises_for_missing_manifest():
    registry = ModelRegistry()

    missing_manifest = Path("missing-manifest.yaml")

    try:
        registry.load_from_manifest_file(str(missing_manifest))
    except FileNotFoundError as exc:
        assert str(missing_manifest) in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing manifest")
