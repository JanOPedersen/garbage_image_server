from flask_smorest import Blueprint
from app.services.model_registry import model_registry

blp = Blueprint("models", "models", url_prefix="/models")


@blp.route("")
@blp.response(200)
def list_models():
    return {
        "models": model_registry.describe_models()
    }