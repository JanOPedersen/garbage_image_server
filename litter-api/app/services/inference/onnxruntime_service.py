import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.services.inference.base import BaseModelService
from app.services.inference.postprocessing import serialize_detections
from app.services.inference.preprocessing import decode_image

log = logging.getLogger(__name__)


class ONNXRuntimeService(BaseModelService):
    def __init__(self, manifest: dict[str, Any]) -> None:
        super().__init__(manifest)
        self.session = None
        self.input_name: str | None = None
        self.output_names: list[str] = []
        self.output_map: dict[str, str] = {}

    def _get_onnxruntime(self) -> Any:
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise RuntimeError(
                "Failed to import onnxruntime. Ensure onnxruntime is installed in the active environment."
            ) from exc
        return ort

    def _resolve_weights_path(self, raw_path: str) -> Path:
        normalized = raw_path.replace("\\", "/")
        artifacts_dir = os.getenv("MODEL_ARTIFACTS_DIR")

        log.info("Resolving model path: '%s'", raw_path)
        log.info("  MODEL_ARTIFACTS_DIR = %s", artifacts_dir or "(not set)")

        if not artifacts_dir:
            raise FileNotFoundError(
                "Model file not found. "
                f"Configured path: '{raw_path}'. "
                "MODEL_ARTIFACTS_DIR is not set."
            )

        candidate = (Path(artifacts_dir) / normalized.lstrip("/")).resolve()
        log.info("  Candidate path: %s", candidate)
        log.info("  Exists: %s", candidate.exists())

        if candidate.exists():
            return candidate

        raise FileNotFoundError(
            "Model file not found. "
            f"Configured path: '{raw_path}'. "
            f"Checked: ['{candidate}']. "
            "Ensure MODEL_ARTIFACTS_DIR is set correctly or use a relative path like "
            "'cigarette-butt/model_final.onnx'."
        )

    def _default_output_map(self, output_names: list[str]) -> dict[str, str]:
        lowered = {name.lower(): name for name in output_names}

        def first_match(candidates: list[str]) -> str | None:
            for candidate in candidates:
                for key, original in lowered.items():
                    if candidate in key:
                        return original
            return None

        mapping: dict[str, str] = {}
        boxes_name = first_match(["boxes", "bbox", "bboxes"])
        scores_name = first_match(["scores", "score", "confidence"])
        classes_name = first_match(["classes", "class_ids", "labels", "label"])
        masks_name = first_match(["masks", "mask", "segmentation"])

        if boxes_name:
            mapping["boxes"] = boxes_name
        if scores_name:
            mapping["scores"] = scores_name
        if classes_name:
            mapping["classes"] = classes_name
        if masks_name:
            mapping["masks"] = masks_name

        return mapping

    def _resize_if_needed(self, image: np.ndarray) -> np.ndarray:
        input_size = self.manifest.get("input_size")
        if not input_size or len(input_size) != 2:
            return image

        target_h, target_w = int(input_size[0]), int(input_size[1])
        if image.shape[0] == target_h and image.shape[1] == target_w:
            return image

        from PIL import Image

        return np.array(
            Image.fromarray(image).resize((target_w, target_h), resample=Image.BILINEAR)
        )

    def _cast_and_normalize(self, image: np.ndarray, input_dtype: str) -> np.ndarray:
        if input_dtype.startswith("float"):
            tensor = image.astype(np.float32)
            if bool(self.manifest.get("normalize", True)):
                tensor = tensor / 255.0

            mean = self.manifest.get("mean")
            std = self.manifest.get("std")
            if mean is not None and std is not None:
                mean_arr = np.array(mean, dtype=np.float32).reshape((1, 1, 3))
                std_arr = np.array(std, dtype=np.float32).reshape((1, 1, 3))
                tensor = (tensor - mean_arr) / std_arr

            if input_dtype == "float16":
                tensor = tensor.astype(np.float16)

            return tensor

        if input_dtype == "uint8":
            return image.astype(np.uint8)

        raise ValueError(f"Unsupported input_dtype: {input_dtype}")

    @staticmethod
    def _apply_layout(tensor: np.ndarray, input_format: str) -> np.ndarray:
        if input_format == "nchw":
            tensor = np.transpose(tensor, (2, 0, 1))
            return np.expand_dims(tensor, axis=0)
        if input_format == "nhwc":
            return np.expand_dims(tensor, axis=0)
        raise ValueError(f"Unsupported input_format: {input_format}")

    def _prepare_input(self, image_rgb: np.ndarray) -> np.ndarray:
        image = self._resize_if_needed(image_rgb)

        if self.manifest.get("channel_order", "rgb").lower() == "bgr":
            image = image[..., ::-1]

        input_format = str(self.manifest.get("input_format", "nchw")).lower()
        input_dtype = str(self.manifest.get("input_dtype", "float32")).lower()
        tensor = self._cast_and_normalize(image, input_dtype)
        tensor = self._apply_layout(tensor, input_format)

        return np.ascontiguousarray(tensor)

    def _get_model_input_size(self) -> tuple[int, int]:
        input_size = self.manifest.get("input_size")
        if input_size and len(input_size) == 2:
            return int(input_size[0]), int(input_size[1])

        if self.session is None:
            raise RuntimeError("Model session is not initialized")

        input_shape = self.session.get_inputs()[0].shape
        if len(input_shape) >= 4:
            height = input_shape[2] if isinstance(input_shape[2], int) else 1
            width = input_shape[3] if isinstance(input_shape[3], int) else 1
            return int(height), int(width)

        raise RuntimeError("Unable to infer ONNX input size")

    @staticmethod
    def _clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
        boxes[:, 0] = np.clip(boxes[:, 0], 0, width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, height)
        return boxes

    @staticmethod
    def _compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        union = box_area + boxes_area - inter
        union = np.where(union <= 0.0, 1e-6, union)
        return inter / union

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> list[int]:
        if boxes.size == 0:
            return []

        iou_threshold = float(self.manifest.get("nms_iou_threshold", 0.45))
        max_detections = int(self.manifest.get("max_detections", 100))
        order = scores.argsort()[::-1]
        keep: list[int] = []

        while order.size > 0 and len(keep) < max_detections:
            current = int(order[0])
            keep.append(current)
            if order.size == 1:
                break

            ious = self._compute_iou(boxes[current], boxes[order[1:]])
            remaining = np.nonzero(ious <= iou_threshold)[0]
            order = order[remaining + 1]

        return keep

    def _decode_ultralytics_yolo(
        self,
        raw_output: Any,
        image_meta: dict[str, Any],
        score_threshold: float | None,
    ) -> tuple[list[list[float]], list[float], list[int], list[np.ndarray] | None]:
        predictions = np.asarray(raw_output)
        if predictions.size == 0:
            return [], [], [], None

        predictions = np.squeeze(predictions)
        if predictions.ndim != 2:
            return [], [], [], None

        if predictions.shape[0] <= predictions.shape[1]:
            predictions = predictions.T

        if predictions.shape[1] < 5:
            return [], [], [], None

        boxes_xywh = predictions[:, :4].astype(np.float32)
        class_scores = predictions[:, 4:].astype(np.float32)

        if class_scores.ndim == 1:
            class_scores = class_scores.reshape(-1, 1)

        scores = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1).astype(np.int64)
        threshold = float(
            score_threshold
            if score_threshold is not None
            else self.manifest.get("default_score_threshold", 0.5)
        )

        keep = np.nonzero(scores >= threshold)[0]
        if keep.size == 0:
            return [], [], [], None

        boxes_xywh = boxes_xywh[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        model_h, model_w = self._get_model_input_size()
        image_w = int(image_meta["width"])
        image_h = int(image_meta["height"])
        scale_x = image_w / float(model_w)
        scale_y = image_h / float(model_h)

        boxes_xyxy = np.empty_like(boxes_xywh)
        boxes_xyxy[:, 0] = (boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0) * scale_x
        boxes_xyxy[:, 1] = (boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0) * scale_y
        boxes_xyxy[:, 2] = (boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0) * scale_x
        boxes_xyxy[:, 3] = (boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0) * scale_y
        boxes_xyxy = self._clip_boxes(boxes_xyxy, image_w, image_h)

        nms_keep = self._nms(boxes_xyxy, scores)
        boxes_xyxy = boxes_xyxy[nms_keep]
        scores = scores[nms_keep]
        class_ids = class_ids[nms_keep]

        return boxes_xyxy.tolist(), scores.tolist(), class_ids.tolist(), None

    def _decode_standard_outputs(
        self,
        outputs_by_name: dict[str, Any],
        score_threshold: float | None,
        return_masks: bool,
    ) -> tuple[list[list[float]], list[float], list[int], list[np.ndarray] | None]:
        boxes_raw = outputs_by_name.get(self.output_map.get("boxes", ""))
        scores_raw = outputs_by_name.get(self.output_map.get("scores", ""))
        classes_raw = outputs_by_name.get(self.output_map.get("classes", ""))
        masks_raw = outputs_by_name.get(self.output_map.get("masks", ""))

        pred_boxes = self._to_2d_boxes(boxes_raw)
        scores = self._to_1d(scores_raw, np.float32)
        pred_classes = self._to_1d(classes_raw, np.int64)

        count = len(pred_boxes)
        if not scores:
            scores = [1.0] * count
        if not pred_classes:
            pred_classes = [0] * count

        count = min(len(pred_boxes), len(scores), len(pred_classes))
        pred_boxes = pred_boxes[:count]
        scores = scores[:count]
        pred_classes = pred_classes[:count]

        kept = list(range(count))
        if score_threshold is not None:
            threshold = float(score_threshold)
            kept = [i for i, s in enumerate(scores) if s >= threshold]
            pred_boxes = [pred_boxes[i] for i in kept]
            scores = [scores[i] for i in kept]
            pred_classes = [pred_classes[i] for i in kept]

        masks = None
        if return_masks and masks_raw is not None:
            masks = self._extract_masks(masks_raw)
            if score_threshold is not None:
                masks = [masks[i] for i in kept if i < len(masks)]

        return pred_boxes, scores, pred_classes, masks

    def _decode_outputs(
        self,
        outputs_by_name: dict[str, Any],
        image_meta: dict[str, Any],
        score_threshold: float | None,
        return_masks: bool,
    ) -> tuple[list[list[float]], list[float], list[int], list[np.ndarray] | None]:
        decoder = str(self.manifest.get("output_decoder", "")).lower()
        if decoder == "ultralytics_yolo":
            raw_name = self.output_map.get("raw_detections") or (self.output_names[0] if self.output_names else "")
            raw_output = outputs_by_name.get(raw_name)
            return self._decode_ultralytics_yolo(raw_output, image_meta, score_threshold)

        return self._decode_standard_outputs(outputs_by_name, score_threshold, return_masks)

    @staticmethod
    def _to_2d_boxes(arr: Any) -> list[list[float]]:
        if arr is None:
            return []
        a = np.asarray(arr)
        if a.size == 0:
            return []
        a = np.squeeze(a)
        if a.ndim == 1 and a.shape[0] == 4:
            a = a.reshape(1, 4)
        if a.ndim != 2 or a.shape[-1] < 4:
            return []
        return a[:, :4].astype(np.float32).tolist()

    @staticmethod
    def _to_1d(arr: Any, dtype: Any) -> list[Any]:
        if arr is None:
            return []
        a = np.asarray(arr)
        if a.size == 0:
            return []
        a = np.squeeze(a)
        if a.ndim == 0:
            a = a.reshape(1)
        return a.astype(dtype).tolist()

    def _extract_masks(self, arr: Any) -> list[np.ndarray]:
        if arr is None:
            return []
        masks = np.asarray(arr)
        if masks.size == 0:
            return []

        masks = np.squeeze(masks)
        if masks.ndim == 2:
            masks = np.expand_dims(masks, axis=0)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0, :, :]
        if masks.ndim != 3:
            return []

        return [(mask > 0.5).astype(np.uint8) for mask in masks]

    def _fake_mask_tag(self, mask: np.ndarray) -> str:
        area = int(mask.sum())
        if mask.ndim == 2:
            return f"mask_{mask.shape[0]}x{mask.shape[1]}_{area}"
        return f"mask_unknown_{area}"

    def load(self) -> None:
        t_load_start = time.perf_counter()
        log.info("[%s] Starting ONNX Runtime load", self.model_id)

        weights_path = self._resolve_weights_path(str(self.manifest["weights_path"]))
        log.info("[%s] Resolved model path: %s", self.model_id, weights_path)

        ort = self._get_onnxruntime()
        session_options = ort.SessionOptions()

        intra_threads = self.manifest.get("intra_op_num_threads")
        inter_threads = self.manifest.get("inter_op_num_threads")
        if intra_threads is not None:
            session_options.intra_op_num_threads = int(intra_threads)
        if inter_threads is not None:
            session_options.inter_op_num_threads = int(inter_threads)

        providers = self.manifest.get("ort_providers")
        if not providers:
            providers = ["CPUExecutionProvider"]

        try:
            self.session = ort.InferenceSession(
                str(weights_path),
                sess_options=session_options,
                providers=providers,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize ONNX Runtime session for model '{self.model_id}'. "
                f"Resolved model path: {weights_path}"
            ) from exc

        inputs = self.session.get_inputs()
        if not inputs:
            raise RuntimeError(f"Model '{self.model_id}' has no ONNX inputs")

        self.input_name = str(self.manifest.get("input_name") or inputs[0].name)
        self.output_names = [o.name for o in self.session.get_outputs()]

        manifest_output_map = self.manifest.get("output_map") or {}
        if manifest_output_map:
            self.output_map = {k: v for k, v in manifest_output_map.items() if v in self.output_names}
        else:
            self.output_map = self._default_output_map(self.output_names)

        self.loaded = True
        log.info(
            "[%s] ONNX Runtime load complete in %.2f ms (input=%s, outputs=%s)",
            self.model_id,
            (time.perf_counter() - t_load_start) * 1000,
            self.input_name,
            self.output_names,
        )

    def warmup(self) -> None:
        if not self.loaded or self.session is None or self.input_name is None:
            raise RuntimeError("Model must be loaded before warmup")

        target_h, target_w = 256, 256
        input_size = self.manifest.get("input_size")
        if input_size and len(input_size) == 2:
            target_h, target_w = int(input_size[0]), int(input_size[1])

        dummy_image = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        warmup_input = self._prepare_input(dummy_image)

        t_warmup_start = time.perf_counter()
        _ = self.session.run(self.output_names or None, {self.input_name: warmup_input})
        log.info(
            "[%s] Warmup complete in %.2f ms",
            self.model_id,
            (time.perf_counter() - t_warmup_start) * 1000,
        )

    def predict(
        self,
        image_bytes: bytes,
        filename: str | None = None,
        score_threshold: float | None = None,
        return_masks: bool = False,
    ) -> dict[str, Any]:
        if not self.loaded or self.session is None or self.input_name is None:
            raise RuntimeError(f"Model {self.model_id} not loaded")

        t0 = time.perf_counter()
        image_np, image_meta = decode_image(image_bytes)
        decode_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        input_tensor = self._prepare_input(image_np)
        preprocess_ms = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        raw_outputs = self.session.run(self.output_names or None, {self.input_name: input_tensor})
        forward_ms = (time.perf_counter() - t2) * 1000

        t3 = time.perf_counter()
        outputs_by_name = dict(zip(self.output_names, raw_outputs))
        pred_boxes, scores, pred_classes, masks = self._decode_outputs(
            outputs_by_name=outputs_by_name,
            image_meta=image_meta,
            score_threshold=score_threshold,
            return_masks=return_masks,
        )

        label_map = self.manifest.get("labels", [])
        labels = [
            label_map[class_idx] if class_idx < len(label_map) else str(class_idx)
            for class_idx in pred_classes
        ]

        masks_rle = None
        if return_masks and masks is not None:
            masks_rle = [self._fake_mask_tag(mask) for mask in masks[: len(pred_boxes)]]

        detections = serialize_detections(
            boxes=pred_boxes,
            scores=scores,
            labels=labels,
            masks_rle=masks_rle,
        )
        postprocess_ms = (time.perf_counter() - t3) * 1000

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
