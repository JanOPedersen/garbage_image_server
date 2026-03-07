from typing import Any


def serialize_detections(
    boxes: list[list[float]],
    scores: list[float],
    labels: list[str],
    masks_rle: list[str | None] | None = None,
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        detections.append(
            {
                "label": label,
                "score": float(score),
                "bbox_xyxy": [float(x) for x in box],
                "mask_rle": masks_rle[idx] if masks_rle else None,
            }
        )
    return detections