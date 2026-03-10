import time
from typing import Any

import os

from app.services.inference.base import BaseModelService
from app.services.inference.preprocessing import decode_image
from app.services.inference.postprocessing import serialize_detections


class Detectron2Service(BaseModelService):
    def __init__(self, manifest: dict[str, Any]) -> None:
        super().__init__(manifest)
        self.predictor = None

    def load(self) -> None:
        model_zoo_config = self.manifest["model_zoo_config"]
        weights_path = self.manifest["weights_path"]

        try:
            from detectron2 import model_zoo
            from detectron2.config import get_cfg
        except Exception as exc:
            raise RuntimeError(
                "Failed to import detectron2. Ensure detectron2 is installed and dependency "
                "versions are compatible (e.g. detectron2 v0.6 requires Pillow 9.x)."
            ) from exc

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model_zoo_config))

        cfg.MODEL.DEVICE = self.manifest.get("device", os.getenv("DEVICE", "cpu"))
        cfg.MODEL.WEIGHTS = str(weights_path)

        cfg.MODEL.ROI_HEADS.NUM_CLASSES = int(self.manifest["num_classes"])
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(
            self.manifest.get("default_score_threshold", 0.5)
        )

        self.predictor = "stub"
        self.loaded = True

    def warmup(self) -> None:
        # Run one dummy inference once the real predictor exists
        if not self.loaded:
            raise RuntimeError("Model must be loaded before warmup")

    def predict(
        self,
        image_bytes: bytes,
        score_threshold: float | None = None,
        return_masks: bool = False,
    ) -> dict[str, Any]:
        if not self.loaded:
            raise RuntimeError(f"Model {self.model_id} not loaded")

        t0 = time.perf_counter()
        _, image_meta = decode_image(image_bytes)
        decode_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        # Add resizing / normalization here if needed
        preprocess_ms = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()

        # ---- STUB INFERENCE ----
        # Replace with actual Detectron2 outputs
        boxes = [[10, 20, 40, 60]]
        scores = [0.97]
        labels = ["cigarette_butt"]
        masks_rle = [None] if return_masks else None
        # ------------------------

        forward_ms = (time.perf_counter() - t2) * 1000

        t3 = time.perf_counter()
        detections = serialize_detections(
            boxes=boxes,
            scores=scores,
            labels=labels,
            masks_rle=masks_rle,
        )
        postprocess_ms = (time.perf_counter() - t3) * 1000

        return {
            "model_id": self.model_id,
            "image": image_meta,
            "timing_ms": {
                "decode": round(decode_ms, 2),
                "preprocess": round(preprocess_ms, 2),
                "forward": round(forward_ms, 2),
                "postprocess": round(postprocess_ms, 2),
                "total": 0.0,
            },
            "detections": detections,
        }