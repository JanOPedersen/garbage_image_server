import os
from app import create_app

app = create_app()

if __name__ == "__main__":
    host = os.getenv("FLASK_RUN_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_RUN_PORT", "8000"))

    #debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    #app.run(host=host, port=port, debug=debug)

    app.run(
        host=host,
        port=port,
        debug=False,
        use_reloader=False,
    )