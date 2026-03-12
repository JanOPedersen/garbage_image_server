import logging
from pathlib import Path
import yaml

from app.services.inference.detectron2_service import Detectron2Service

log = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, Detectron2Service] = {}
        self._default_model_id: str | None = None

    def load_from_manifest_file(self, manifest_file: str) -> None:
        log.info("Loading manifest: %s", manifest_file)
        path = Path(manifest_file)
        if not path.exists():
            log.error("Manifest file not found: %s", manifest_file)
            raise FileNotFoundError(f"Manifest file not found: {manifest_file}")

        with path.open("r", encoding="utf-8") as f:
            manifest = yaml.safe_load(f)

        model_id = manifest["model_id"]
        log.info("Manifest loaded — model_id: %s", model_id)
        log.info("  weights_path: %s", manifest.get("weights_path"))
        log.info("  device:       %s", manifest.get("device", "(not set, will use DEVICE env)"))

        service = Detectron2Service(manifest)
        log.info("Loading model weights...")
        service.load()
        log.info("Weights loaded — running warmup...")
        service.warmup()
        log.info("Warmup complete — model '%s' is ready", model_id)

        self._models = {model_id: service}
        self._default_model_id = model_id

    def get(self, model_id: str):
        return self._models.get(model_id)

    @property
    def default_model_id(self) -> str | None:
        return self._default_model_id

    def is_ready(self) -> bool:
        return len(self._models) > 0 and all(m.loaded for m in self._models.values())

    def list_models(self) -> list[str]:
        return list(self._models.keys())

    def describe_models(self) -> list[dict]:
        return [m.metadata() for m in self._models.values()]


model_registry = ModelRegistry()