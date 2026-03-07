import time
from typing import Any

from app.services.inference.base import BaseModelService
from app.services.inference.preprocessing import decode_image
from app.services.inference.postprocessing import serialize_detections


class Detectron2Service(BaseModelService):
    def __init__(self, manifest: dict[str, Any]) -> None:
        super().__init__(manifest)
        self.predictor = None

    def load(self) -> None:
        # Replace this block with real Detectron2 loading
        # Example later:
        # - build cfg
        # - cfg.MODEL.WEIGHTS = ...
        # - cfg.MODEL.DEVICE = ...
        # - self.predictor = DefaultPredictor(cfg)
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
        image_np, image_meta = decode_image(image_bytes)
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