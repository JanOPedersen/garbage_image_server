import io
import time
import uuid

from flask import current_app, request
from flask_smorest import Blueprint, abort

from app.schemas.predict import PredictQuerySchema, PredictResponseSchema
from app.services.model_registry import model_registry

blp = Blueprint("predict", "predict", url_prefix="")


@blp.route("/predict")
@blp.arguments(PredictQuerySchema, location="query")
@blp.response(200, PredictResponseSchema)
def predict(args):
    if "image" not in request.files:
        abort(400, message="Missing image file in multipart form field 'image'")

    image_file = request.files["image"]
    if image_file.filename == "":
        abort(400, message="Empty filename")

    model_id = args.get("model_id") or model_registry.default_model_id
    if model_id is None:
        abort(503, message="No default model is configured")

    service = model_registry.get(model_id)
    if service is None:
        abort(404, message=f"Unknown model_id: {model_id}")

    image_bytes = image_file.read()
    if not image_bytes:
        abort(400, message="Uploaded image is empty")
    filename = image_file.filename

    request_id = f"req_{uuid.uuid4().hex[:12]}"
    started = time.perf_counter()

    result = service.predict(
        image_bytes=image_bytes,
        filename=filename,
        score_threshold=args.get("score_threshold"),
        return_masks=args.get("return_masks", False),
    )

    result["request_id"] = request_id
    result["timing_ms"]["total"] = round((time.perf_counter() - started) * 1000, 2)
    return result