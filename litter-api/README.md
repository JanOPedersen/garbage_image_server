Below is a concrete starter layout that keeps the API thin, the model code isolated, and the future async path easy to add.

## Folder tree

```text
litter-api/
├─ app/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ extensions.py
│  ├─ logging_config.py
│  │
│  ├─ api/
│  │  ├─ __init__.py
│  │  ├─ health.py
│  │  ├─ models.py
│  │  └─ predict.py
│  │
│  ├─ schemas/
│  │  ├─ __init__.py
│  │  ├─ common.py
│  │  └─ predict.py
│  │
│  ├─ services/
│  │  ├─ __init__.py
│  │  ├─ model_registry.py
│  │  └─ inference/
│  │     ├─ __init__.py
│  │     ├─ base.py
│  │     ├─ detectron2_service.py
│  │     ├─ preprocessing.py
│  │     └─ postprocessing.py
│  │
│  └─ domain/
│     ├─ __init__.py
│     └─ manifests.py
│
├─ models/
│  └─ manifests/
│     └─ cigarette-butt-v1.yaml
│
├─ scripts/
│  └─ warmup.py
│
├─ tests/
│  ├─ test_health.py
│  └─ test_predict.py
│
├─ .env.example
├─ .gitignore
├─ Dockerfile
├─ docker-compose.yml
├─ gunicorn.conf.py
├─ requirements.txt
├─ run.py
└─ README.md
```

------

## What goes where

- `app/api/`: Flask-Smorest blueprints only
- `app/schemas/`: Marshmallow request/response schemas
- `app/services/inference/`: model loading, predict, preprocess, postprocess
- `app/services/model_registry.py`: maps `model_id` to a service instance
- `models/manifests/`: model metadata and paths
- `run.py`: local entrypoint
- `gunicorn.conf.py`: production web server settings

This is enough for:

- one fast sync model now
- multiple model IDs later
- future `/jobs` worker without refactoring the whole app

------

# Minimal code skeleton

## `requirements.txt`

```txt
flask==3.1.0
flask-smorest==0.45.0
marshmallow==3.26.1
gunicorn==23.0.0
python-dotenv==1.0.1
PyYAML==6.0.2
Pillow==11.1.0
numpy==2.2.3

# Add these in your real project once your runtime is pinned
# torch==...
# torchvision==...
# detectron2 @ git+https://github.com/facebookresearch/detectron2.git@...
```

For the first pass, you can even mock the inference service until the API shape is stable.

------

## `run.py`

```python
from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
```

------

## `app/__init__.py`

```python
from flask import Flask

from app.api.health import blp as HealthBlueprint
from app.api.models import blp as ModelsBlueprint
from app.api.predict import blp as PredictBlueprint
from app.config import Config
from app.extensions import api
from app.logging_config import configure_logging
from app.services.model_registry import model_registry


def create_app(config_object: type[Config] = Config) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_object)

    configure_logging()

    api.init_app(app)
    api.register_blueprint(HealthBlueprint)
    api.register_blueprint(ModelsBlueprint)
    api.register_blueprint(PredictBlueprint)

    # Load models once at startup
    with app.app_context():
        model_registry.load_from_manifest_dir(app.config["MODEL_MANIFEST_DIR"])

    return app
```

------

## `app/config.py`

```python
import os


class Config:
    API_TITLE = "Litter Detection API"
    API_VERSION = "v1"
    OPENAPI_VERSION = "3.0.3"
    OPENAPI_URL_PREFIX = "/"
    OPENAPI_SWAGGER_UI_PATH = "/docs"
    OPENAPI_SWAGGER_UI_URL = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/"

    PROPAGATE_EXCEPTIONS = True

    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 8 * 1024 * 1024))
    MODEL_MANIFEST_DIR = os.getenv("MODEL_MANIFEST_DIR", "models/manifests")
    DEFAULT_MODEL_ID = os.getenv("DEFAULT_MODEL_ID", "cigarette-butt-v1")
    DEVICE = os.getenv("DEVICE", "cpu")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
```

------

## `app/extensions.py`

```python
from flask_smorest import Api

api = Api()
```

------

## `app/logging_config.py`

```python
import logging
import os


def configure_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
```

------

## `app/api/health.py`

```python
from flask_smorest import Blueprint
from app.services.model_registry import model_registry

blp = Blueprint("health", "health", url_prefix="")


@blp.route("/healthz")
@blp.response(200)
def healthz():
    return {"status": "ok"}


@blp.route("/readyz")
@blp.response(200)
def readyz():
    return {
        "status": "ready" if model_registry.is_ready() else "not_ready",
        "loaded_models": model_registry.list_models(),
    }
```

------

## `app/api/models.py`

```python
from flask_smorest import Blueprint
from app.services.model_registry import model_registry

blp = Blueprint("models", "models", url_prefix="/models")


@blp.route("")
@blp.response(200)
def list_models():
    return {
        "models": model_registry.describe_models()
    }
```

------

## `app/schemas/common.py`

```python
from marshmallow import Schema, fields


class TimingSchema(Schema):
    decode = fields.Float(required=True)
    preprocess = fields.Float(required=True)
    forward = fields.Float(required=True)
    postprocess = fields.Float(required=True)
    total = fields.Float(required=True)
```

------

## `app/schemas/predict.py`

```python
from marshmallow import Schema, fields


class PredictQuerySchema(Schema):
    model_id = fields.String(required=False)
    score_threshold = fields.Float(required=False, load_default=None)
    return_masks = fields.Boolean(required=False, load_default=False)


class DetectionSchema(Schema):
    label = fields.String(required=True)
    score = fields.Float(required=True)
    bbox_xyxy = fields.List(fields.Float(), required=True)
    mask_rle = fields.String(allow_none=True)


class PredictResponseSchema(Schema):
    request_id = fields.String(required=True)
    model_id = fields.String(required=True)
    image = fields.Dict(required=True)
    timing_ms = fields.Dict(required=True)
    detections = fields.List(fields.Nested(DetectionSchema), required=True)
```

------

## `app/api/predict.py`

```python
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

    model_id = args.get("model_id") or current_app.config["DEFAULT_MODEL_ID"]
    service = model_registry.get(model_id)
    if service is None:
        abort(404, message=f"Unknown model_id: {model_id}")

    image_bytes = image_file.read()
    if not image_bytes:
        abort(400, message="Uploaded image is empty")

    request_id = f"req_{uuid.uuid4().hex[:12]}"
    started = time.perf_counter()

    result = service.predict(
        image_bytes=image_bytes,
        score_threshold=args.get("score_threshold"),
        return_masks=args.get("return_masks", False),
    )

    result["request_id"] = request_id
    result["timing_ms"]["total"] = round((time.perf_counter() - started) * 1000, 2)
    return result
```

------

## `app/services/inference/base.py`

```python
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
```

------

## `app/services/inference/preprocessing.py`

```python
import io
from PIL import Image
import numpy as np


def decode_image(image_bytes: bytes) -> tuple[np.ndarray, dict]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    arr = np.array(image)
    meta = {"width": width, "height": height}
    return arr, meta
```

------

## `app/services/inference/postprocessing.py`

```python
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
```

------

## `app/services/inference/detectron2_service.py`

This version is intentionally minimal. First with a stub, then you replace the marked section with your real Detectron2 code.

```python
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
```

------

## `app/services/model_registry.py`

```python
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
```

------

## `models/manifests/cigarette-butt-v1.yaml`

```yaml
model_id: cigarette-butt-v1
version: 2026-03-01
task_type: instance_segmentation
framework: detectron2
device: cpu
weights_path: /app/model_artifacts/cigarette-butt/model_final.pth
config_path: /app/model_artifacts/cigarette-butt/config.yaml
labels:
  - cigarette_butt
default_score_threshold: 0.5
max_image_side: 2048
sync_allowed: true
```

Later, you can add:

- `olm-multi-v1.yaml`
- `heap-analysis-v1.yaml`

without touching the API routes.

------

## `gunicorn.conf.py`

```python
bind = "0.0.0.0:5000"
workers = 1
threads = 4
timeout = 120
graceful_timeout = 30
accesslog = "-"
errorlog = "-"
loglevel = "info"
```

For GPU inference, start with `workers = 1`. Multiple workers can multiply model memory usage badly.

------

# Docker

## `Dockerfile`

This is a CPU-safe starter. After your API shape is stable, make a GPU-specific Dockerfile or switch the base image.

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "run:app", "-c", "gunicorn.conf.py"]
```

When you move to real Detectron2 + CUDA, you will probably want a different base image and pinned torch/detectron2 versions.

------

## `docker-compose.yml`

```yaml
version: "3.9"

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: litter-api
    ports:
      - "5000:5000"
    environment:
      LOG_LEVEL: INFO
      DEVICE: cpu
      DEFAULT_MODEL_ID: cigarette-butt-v1
      MODEL_MANIFEST_DIR: models/manifests
      MAX_CONTENT_LENGTH: 8388608
    volumes:
      - .:/app
    command: python run.py
```

For local debugging, `python run.py` is nicer than Gunicorn.

For production-like local testing, run:

```bash
docker compose run --service-ports api gunicorn run:app -c gunicorn.conf.py
```

------

# Minimal tests

## `tests/test_health.py`

```python
from app import create_app


def test_healthz():
    app = create_app()
    client = app.test_client()

    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"
```

------

# How easy this is to debug in VS Code

Pretty easy.

## Local debugging without Docker: easiest

Best for:

- breakpoints in routes
- stepping through preprocessing/postprocessing
- quick iteration

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug Flask app",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/run.py",
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}",
      "env": {
        "LOG_LEVEL": "DEBUG",
        "DEVICE": "cpu",
        "DEFAULT_MODEL_ID": "cigarette-butt-v1",
        "MODEL_MANIFEST_DIR": "models/manifests"
      },
      "justMyCode": false
    }
  ]
}
```

That gives you the smoothest experience.

------

## Debugging with Docker: still manageable

Good for:

- matching deployment environment
- dependency issues
- import/path issues
- container-only bugs

A minimal `.vscode/launch.json` for attach debugging is possible, but I would not start there. For day-to-day development:

- debug locally outside Docker
- validate final behavior inside Docker

That is the least painful workflow.

------

## Debugging Detectron2-specific issues

This structure helps because you can set breakpoints in exactly these places:

- `decode_image`
- preprocessing
- real predictor call
- postprocessing
- route serialization

So when something breaks, you can quickly answer:

- did the upload decode correctly?
- is the model loaded?
- did inference return empty instances?
- did mask serialization fail?

That is much easier than burying everything inside one Flask route.

------

# What to replace first with real model code

Once the stub is running, the first real upgrade should be only inside:

```text
app/services/inference/detectron2_service.py
```

You replace:

- `load()`
- `warmup()`
- `predict()`

Everything else can stay mostly unchanged.

That is the main reason this structure scales well.

------

# Suggested next steps for you

1. Run this skeleton with the stub model first.
2. Confirm `/docs`, `/healthz`, `/readyz`, `/models`, `/predict`.
3. Replace the stub inference service with your Detectron2 predictor.
4. Measure decode / preprocess / forward / postprocess timing.
5. Only then decide whether to split out async workers.

If you want, I’ll turn this into a **real Detectron2-integrated version** of `detectron2_service.py` using `DefaultPredictor`, config loading, and mask extraction.