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