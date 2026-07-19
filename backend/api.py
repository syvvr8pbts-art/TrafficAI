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

from typing import Any, Optional

from .config import BackendConfig, get_config
from .pipeline import AnalyticsPipeline


def create_app(
    pipeline: Optional[AnalyticsPipeline] = None,
    config: Optional[BackendConfig] = None,
    system: Optional[Any] = None,
    processor: Optional[Any] = None,
):
    """Application factory: build and return a configured Flask app.

    Args:
        pipeline: the analytics dispatcher (used if ``system`` is not given).
        config: backend configuration (server + detector settings).
        system: a TrafficSystem exposing ``snapshot()``/``full_state()`` and
            ``pipeline`` for the live state endpoints.
        processor: an optional VideoProcessor providing the annotated MJPEG
            stream for the dashboard's live video.
    """
    from flask import Flask, Response, jsonify, render_template  # lazy import

    config = config or get_config()
    # Prefer the full traffic system; fall back to a bare pipeline.
    pipeline = (system.pipeline if system is not None else pipeline) or AnalyticsPipeline()

    app = Flask(__name__)  # templates/ and static/ resolve under backend/
    # Stash shared objects on the app so future blueprints can reach them.
    app.config["backend_config"] = config
    app.config["pipeline"] = pipeline
    app.config["system"] = system
    app.config["processor"] = processor

    # -- existing endpoints (unchanged behavior) ---------------------------

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
            "endpoints": ["/", "/health", "/signal", "/state", "/video_feed", "/dashboard"],
        })

    @app.get("/signal")
    def signal():
        """Live signal-intelligence state: current signal, lane, countdown,
        recommended green, intersection status, and ambulance mode.
        """
        if system is None:
            return jsonify({"error": "signal system not available"}), 503
        return jsonify(system.snapshot())

    # -- dashboard integration endpoints -----------------------------------

    @app.get("/state")
    def state():
        """Full live state (signal + analytics) consumed by the dashboard."""
        if system is None:
            return jsonify({"error": "signal system not available"}), 503
        return jsonify(system.full_state())

    @app.get("/video_feed")
    def video_feed():
        """Processed OpenCV stream as multipart MJPEG."""
        if processor is None:
            return jsonify({"error": "video stream not available"}), 503
        return Response(
            processor.mjpeg_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/dashboard")
    def dashboard():
        """Serve the traffic control center dashboard page."""
        return render_template("dashboard.html")

    return app
