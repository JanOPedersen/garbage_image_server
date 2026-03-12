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
    DEFAULT_MANIFEST_FILE = os.getenv(
        "DEFAULT_MANIFEST_FILE",
        "models/manifests/cigarette-butt-v1.yaml",
    )
    DEVICE = os.getenv("DEVICE", "cpu")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")