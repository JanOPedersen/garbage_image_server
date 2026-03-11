import time
from typing import Any
import importlib

import os
from pathlib import Path

from app.services.inference.base import BaseModelService
from app.services.inference.preprocessing import decode_image
from app.services.inference.postprocessing import serialize_detections


class Detectron2Service(BaseModelService):
    def __init__(self, manifest: dict[str, Any]) -> None:
        super().__init__(manifest)
        self.predictor = None

    def _get_torch(self) -> Any:
        try:
            return importlib.import_module("torch")
        except Exception as exc:
            raise RuntimeError(
                "Failed to import torch. Ensure torch is installed in the active environment."
            ) from exc

    def load(self) -> None:
        model_zoo_config = self.manifest["model_zoo_config"]
        weights_path = self._resolve_weights_path(str(self.manifest["weights_path"]))

        try:
            model_zoo = importlib.import_module("detectron2.model_zoo")
            get_cfg = importlib.import_module("detectron2.config").get_cfg
            default_predictor = importlib.import_module("detectron2.engine").DefaultPredictor
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

        if "min_size_test" in self.manifest:
            cfg.INPUT.MIN_SIZE_TEST = int(self.manifest["min_size_test"])

        if "max_size_test" in self.manifest:
            cfg.INPUT.MAX_SIZE_TEST = int(self.manifest["max_size_test"])

        self.cfg = cfg
        try:
            self.predictor = default_predictor(cfg)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize Detectron2 predictor for model '{self.model_id}'. "
                f"Resolved weights path: {weights_path}"
            ) from exc
        self.loaded = True

    def _resolve_weights_path(self, raw_path: str) -> Path:
        normalized = raw_path.replace("\\", "/")
        artifacts_dir = os.getenv("MODEL_ARTIFACTS_DIR")

        if not artifacts_dir:
            raise FileNotFoundError(
                "Model weights not found. "
                f"Configured path: '{raw_path}'. "
                "MODEL_ARTIFACTS_DIR is not set."
            )

        candidate = (Path(artifacts_dir) / normalized.lstrip("/")).resolve()
        if candidate.exists():
            return candidate

        raise FileNotFoundError(
            "Model weights not found. "
            f"Configured path: '{raw_path}'. "
            f"Checked: ['{candidate}']. "
            "Ensure MODEL_ARTIFACTS_DIR is set correctly or use a relative path like "
            "'cigarette-butt/model_final.pth'."
        )

    def warmup(self) -> None:
        # Run one dummy inference once the real predictor exists
        if not self.loaded:
            raise RuntimeError("Model must be loaded before warmup")
        
        if not self.predictor:
            raise RuntimeError("Predictor is not loaded")

        torch = self._get_torch()

        # Warmup with a small dummy image
        dummy = torch.zeros((256, 256, 3), dtype=torch.uint8).numpy()
        with torch.inference_mode():
            _ = self.predictor(dummy)

    def predict(
        self,
        image_bytes: bytes,
        filename: str | None = None,
        score_threshold: float | None = None,
        return_masks: bool = False,
    ) -> dict[str, Any]:
        if not self.loaded:
            raise RuntimeError(f"Model {self.model_id} not loaded")

        torch = self._get_torch()

        t0 = time.perf_counter()
        image_np, image_meta = decode_image(image_bytes)
        decode_ms = (time.perf_counter() - t0) * 1000
        # No additional preprocessing is currently applied after decode.
        preprocess_ms = 0.0

        t1 = time.perf_counter()
        with torch.inference_mode():
            outputs = self.predictor(image_np)
        forward_ms = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        instances = outputs["instances"].to("cpu")

        pred_boxes = instances.pred_boxes.tensor.tolist() if instances.has("pred_boxes") else []
        scores = instances.scores.tolist() if instances.has("scores") else []
        pred_classes = instances.pred_classes.tolist() if instances.has("pred_classes") else []

        if score_threshold is not None:
            kept = [i for i, s in enumerate(scores) if s >= float(score_threshold)]
            pred_boxes = [pred_boxes[i] for i in kept]
            scores = [scores[i] for i in kept]
            pred_classes = [pred_classes[i] for i in kept]

        label_map = self.manifest.get("labels", [])
        labels = [
            label_map[class_idx] if class_idx < len(label_map) else str(class_idx)
            for class_idx in pred_classes
        ]

        masks_rle = None
        if return_masks and instances.has("pred_masks"):
            masks = instances.pred_masks.numpy()
            if score_threshold is not None:
                masks = [masks[i] for i in kept]
            masks_rle = [self._fake_mask_tag(mask) for mask in masks]

        detections = serialize_detections(
            boxes=pred_boxes,
            scores=scores,
            labels=labels,
            masks_rle=masks_rle,
        )
        postprocess_ms = (time.perf_counter() - t2) * 1000

        if filename:
            image_meta["filename"] = filename

        return {
            "model_id": self.model_id,
            "image": image_meta,
            "timing_ms": {
                "decode": round(decode_ms, 2),
                "preprocess": round(preprocess_ms, 2),
                "forward": round(forward_ms, 2),
                "postprocess": round(postprocess_ms, 2),
                "total": round(decode_ms + preprocess_ms + forward_ms + postprocess_ms, 2),
            },
            "detections": detections,
        }

    def _fake_mask_tag(self, mask: Any) -> str:
        # Lightweight placeholder mask representation until real RLE encoding is added.
        area = int(mask.sum()) if hasattr(mask, "sum") else 0
        shape = getattr(mask, "shape", None)
        if isinstance(shape, tuple) and len(shape) >= 2:
            return f"mask_{shape[0]}x{shape[1]}_{area}"
        return f"mask_unknown_{area}"