from pathlib import Path
import yaml

from app.services.inference.detectron2_service import Detectron2Service


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, Detectron2Service] = {}

    def load_from_manifest_dir(self, manifest_dir: str) -> None:
        path = Path(manifest_dir)
        if not path.exists():
            raise FileNotFoundError(f"Manifest directory not found: {manifest_dir}")

        for manifest_file in path.glob("*.yaml"):
            with manifest_file.open("r", encoding="utf-8") as f:
                manifest = yaml.safe_load(f)

            model_id = manifest["model_id"]
            service = Detectron2Service(manifest)
            service.load()
            service.warmup()
            self._models[model_id] = service

    def get(self, model_id: str):
        return self._models.get(model_id)

    def is_ready(self) -> bool:
        return len(self._models) > 0 and all(m.loaded for m in self._models.values())

    def list_models(self) -> list[str]:
        return list(self._models.keys())

    def describe_models(self) -> list[dict]:
        return [m.metadata() for m in self._models.values()]


model_registry = ModelRegistry()