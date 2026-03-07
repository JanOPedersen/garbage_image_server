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