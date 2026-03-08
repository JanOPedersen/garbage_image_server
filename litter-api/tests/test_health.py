from app import create_app
from app.config import Config


class TestConfig(Config):
    TESTING = True


def test_healthz():
    app = create_app(TestConfig)
    client = app.test_client()

    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"