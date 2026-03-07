from abc import ABC, abstractmethod
from typing import Any


class BaseModelService(ABC):
    def __init__(self, manifest: dict[str, Any]) -> None:
        self.manifest = manifest
        self.model_id = manifest["model_id"]
        self.loaded = False

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def warmup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        image_bytes: bytes,
        score_threshold: float | None = None,
        return_masks: bool = False,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.manifest["model_id"],
            "task_type": self.manifest.get("task_type"),
            "version": self.manifest.get("version"),
            "labels": self.manifest.get("labels", []),
            "loaded": self.loaded,
        }