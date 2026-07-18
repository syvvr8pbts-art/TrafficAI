"""
backend/api.py

HTTP API layer (Flask). Intentionally thin: it does routing and serialization
only -- no detection and no analytics logic. It exposes health/info endpoints
now, and is structured so future analytics endpoints attach to the same app and
read from the ``AnalyticsPipeline`` / its consumers, never from the detector
directly.

Flask is imported lazily inside ``create_app`` so that importing the rest of the
backend (and running the existing detection pipeline) never requires Flask.
"""
from __future__ import annotations

from typing import Optional

from .config import BackendConfig, get_config
from .pipeline import AnalyticsPipeline


def create_app(
    pipeline: Optional[AnalyticsPipeline] = None,
    config: Optional[BackendConfig] = None,
):
    """Application factory: build and return a configured Flask app.

    Args:
        pipeline: the analytics dispatcher whose consumers back the API.
        config: backend configuration (server + detector settings).
    """
    from flask import Flask, jsonify  # lazy import keeps core Flask-free

    config = config or get_config()
    pipeline = pipeline or AnalyticsPipeline()

    app = Flask(__name__)
    # Stash shared objects on the app so future blueprints can reach them.
    app.config["backend_config"] = config
    app.config["pipeline"] = pipeline

    @app.get("/health")
    def health():
        """Liveness probe."""
        return jsonify({"status": "ok"})

    @app.get("/")
    def info():
        """Basic service metadata and current wiring."""
        return jsonify({
            "service": "TrafficAI backend",
            "device": config.detector.device,
            "weights": config.detector.weights_path,
            "registered_consumers": pipeline.consumers(),
            "endpoints": ["/", "/health"],
        })

    # -- Future analytics endpoints attach below, e.g. --------------------
    #   from statistics import StatisticsService
    #   app.register_blueprint(StatisticsService(pipeline).blueprint)
    # ---------------------------------------------------------------------

    return app
