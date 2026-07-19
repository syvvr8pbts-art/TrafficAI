"""
backend/app.py

Composition root and entry point for the backend. It wires the pieces together
(config -> detector -> pipeline -> api) and exposes run modes. This is the ONE
place future analytics modules are registered on the pipeline -- a couple of
lines each -- so adding features never touches ``detector.py``.

Run modes:
    python -m backend.app              # print readiness/wiring info and exit
    python -m backend.app --selftest   # validate the payload contract (no model)
    python -m backend.app --serve      # start the Flask API server
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from typing import Any, Dict, List, Optional

import numpy as np

# Root Detection dataclass, used only to build a synthetic payload in --selftest.
from detector import Detection

from .ambulance import AmbulancePriority
from .clearance import IntersectionClearance
from .config import get_config
from .counter import CANONICAL_CLASSES, VehicleCounter
from .density import TrafficDensity
from .detector import FrameDetector, build_payload
from .logger import DetectionLogger
from .occupancy import LaneOccupancy
from .pipeline import AnalyticsPipeline
from .recommendation import RecommendationEngine
from .signal_controller import SignalController
from .statistics import TrafficStatistics
from .utils import get_logger

logger = get_logger(__name__)


class TrafficSystem:
    """Composition of the full backend pipeline for one intersection.

    It constructs every consumer, cross-wires the signal-intelligence modules
    (clearance and ambulance hold a reference to the controller), registers them
    on the AnalyticsPipeline in order, and keeps the latest payload so the API
    can report live state. This is wiring only -- no module logic lives here.

        Counter -> Density -> Occupancy -> Statistics ->
        Recommendation -> SignalController -> Clearance -> Ambulance -> Logger
    """

    def __init__(self) -> None:
        # Analytics stages (Phase 2, unchanged).
        self.counter = VehicleCounter()
        self.density = TrafficDensity()
        self.occupancy = LaneOccupancy()
        self.statistics = TrafficStatistics()
        # Signal-intelligence stages (Phase 3).
        self.recommendation = RecommendationEngine()
        self.controller = SignalController()
        self.clearance = IntersectionClearance(controller=self.controller)
        self.ambulance = AmbulancePriority(controller=self.controller)
        # Logging last so it records the fully assembled payload.
        self.logger = DetectionLogger()

        self.pipeline = AnalyticsPipeline()
        for consumer in (
            self.counter, self.density, self.occupancy, self.statistics,
            self.recommendation, self.controller, self.clearance, self.ambulance,
            self.logger,
        ):
            self.pipeline.register(consumer)

        self._latest: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch one payload through the pipeline and store it as latest."""
        self.pipeline.dispatch(payload)
        with self._lock:
            self._latest = payload
        return payload

    def snapshot(self) -> Dict[str, Any]:
        """Compact live view for the API (falls back to controller state)."""
        with self._lock:
            p = dict(self._latest)
        signal = p.get("signal") or self.controller.snapshot()
        return {
            "lane": signal.get("lane"),
            "signal": signal.get("state"),
            "countdown": signal.get("countdown"),
            "next_lane": signal.get("next_lane"),
            "recommended_green": p.get("recommended_green"),
            "intersection": (p.get("intersection") or {}).get("status"),
            "ambulance": bool((p.get("ambulance") or {}).get("active", False)),
        }

    def full_state(self) -> Dict[str, Any]:
        """Complete live view for the dashboard: the compact signal snapshot
        plus vehicle counts, density, occupancy, statistics, and lane list.

        This is a read-only projection of the latest payload -- the dashboard
        never computes anything, it only renders these fields.
        """
        with self._lock:
            p = dict(self._latest)
        state = self.snapshot()
        ambulance_info = p.get("ambulance") or {}
        state.update({
            "frame_id": p.get("frame_id"),
            "counts": p.get("counts", {}),
            "total": p.get("total", 0),
            "density": p.get("density"),
            "occupancy": p.get("occupancy"),
            "statistics": p.get("statistics", {}),
            "ambulance_lane": ambulance_info.get("lane"),
            "lanes": list(self.controller.lanes),
        })
        return state


def build_pipeline() -> AnalyticsPipeline:
    """Return a fully wired AnalyticsPipeline (analytics + signal intelligence)."""
    return TrafficSystem().pipeline


# Maps canonical count keys -> the labels used in the terminal summary block.
_SUMMARY_LABELS = [
    ("car", "Cars"), ("bike", "Bikes"), ("auto-rickshaw", "Autos"),
    ("cycle", "Cycles"), ("bus", "Bus"), ("truck", "Truck"),
    ("ambulance", "Ambulance"),
]


def render_terminal_summary(payload: dict) -> str:
    """Format a clean per-frame vehicle-analytics block (Phase 2)."""
    counts = payload.get("counts", {})
    lines = ["=" * 36, f"Frame: {payload.get('frame_id')}"]
    for key, label in _SUMMARY_LABELS:
        lines.append(f"{label:<14}: {counts.get(key, 0)}")
    lines.append(f"{'Total Vehicles':<14}: {payload.get('total', 0)}")
    lines.append(f"{'Traffic Density':<14}: {payload.get('density')}")
    lines.append(f"{'Lane Occupancy':<14}: {payload.get('occupancy')}%")
    lines.append("=" * 36)
    return "\n".join(lines)


def render_signal_summary(payload: dict) -> str:
    """Format the signal-intelligence status block (Phase 3)."""
    signal = payload.get("signal", {})
    intersection = payload.get("intersection", {})
    ambulance = payload.get("ambulance", {})
    amb_state = "ON" if ambulance.get("active") else "OFF"
    return "\n".join([
        "=" * 35,
        f"Lane:\n{signal.get('lane')}", "",
        f"Signal:\n{signal.get('state')}", "",
        f"Countdown:\n{signal.get('countdown')}", "",
        f"Density:\n{payload.get('density')}", "",
        f"Recommended Green:\n{payload.get('recommended_green')} sec", "",
        f"Intersection:\n{intersection.get('status')}", "",
        f"Ambulance:\n{amb_state}", "",
        "=" * 35,
    ])


def run_analysis(source: str, max_frames: int = 0) -> int:
    """Drive a video source through the full detection + analytics + signal pipeline.

    Prints a per-frame signal-status block and logs each frame to CSV. The loop
    (not the detector) enriches the payload with the real frame dimensions so
    occupancy and ROI logic work without changing detector.py.
    """
    import cv2

    cfg = get_config()
    detector = FrameDetector(cfg.detector)
    system = build_traffic_system()

    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    processed = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            payload = detector.process(frame)
            # Enrich payload with real frame size for occupancy + ROI checks.
            payload["frame_height"], payload["frame_width"] = frame.shape[0], frame.shape[1]
            system.process(payload)
            print(render_signal_summary(payload))
            processed += 1
            if max_frames and processed >= max_frames:
                break
    finally:
        cap.release()
    logger.info("Analysis finished: %d frame(s) processed.", processed)
    return 0


def build_traffic_system() -> TrafficSystem:
    """Construct the fully wired traffic system (analytics + signal intelligence)."""
    return TrafficSystem()


def create_api():
    """Build the Flask app wired to a fresh traffic system and config."""
    from .api import create_app  # lazy: only import Flask path when serving
    return create_app(system=build_traffic_system(), config=get_config())


def build_dashboard(source: Optional[str] = None):
    """Wire the full dashboard: a running VideoProcessor feeding a TrafficSystem,
    behind a Flask app. Returns ``(app, processor)``.

    The processor runs the existing pipeline in a background thread; the Flask
    app serves the annotated MJPEG stream and the live JSON state.
    """
    from .api import create_app
    from .streamer import VideoProcessor

    cfg = get_config()
    source = source or cfg.app.video_source
    system = build_traffic_system()
    detector = FrameDetector(cfg.detector)
    processor = VideoProcessor(source, system=system, detector=detector).start()
    app = create_app(system=system, config=cfg, processor=processor)
    return app, processor


def create_detector() -> FrameDetector:
    """Instantiate the frame detector (loads the YOLO model)."""
    return FrameDetector(get_config().detector)


def _selftest() -> int:
    """Validate the detection-payload contract without loading the model."""
    fake: List[Detection] = [
        Detection(xyxy=np.array([10, 20, 110, 220], dtype=np.float32),
                  confidence=0.93, class_id=2, class_name="car"),
        Detection(xyxy=np.array([300, 120, 360, 200], dtype=np.float32),
                  confidence=0.81, class_id=0, class_name="ambulance"),
    ]
    payload = build_payload(frame_id=1, detections=fake)
    print(json.dumps(payload, indent=2))
    assert payload["frame_id"] == 1
    assert payload["detections"][0]["class"] == "car"
    assert payload["detections"][0]["bbox"] == [10.0, 20.0, 110.0, 220.0]
    print("selftest: OK")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="TrafficAI backend")
    parser.add_argument("--serve", action="store_true", help="Start the Flask dashboard + API server")
    parser.add_argument("--selftest", action="store_true", help="Validate the payload contract (no model load)")
    parser.add_argument("--analyze", metavar="SOURCE", help="Run detection + analytics over a video source")
    parser.add_argument("--source", help="Video source for the dashboard stream (defaults to config)")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = no limit)")
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()

    if args.analyze:
        return run_analysis(args.analyze, max_frames=args.max_frames)

    if args.serve:
        cfg = get_config()
        app, processor = build_dashboard(source=args.source)
        logger.info("Dashboard at http://%s:%d/dashboard", cfg.server.host, cfg.server.port)
        try:
            # threaded=True so the MJPEG stream and JSON polling are concurrent.
            app.run(host=cfg.server.host, port=cfg.server.port,
                    debug=cfg.server.debug, threaded=True, use_reloader=False)
        finally:
            processor.stop()
        return 0

    # Default: report how the backend is wired, without loading the model.
    cfg = get_config()
    print(json.dumps({
        "service": "TrafficAI backend",
        "device": cfg.detector.device,
        "weights": cfg.detector.weights_path,
        "server": {"host": cfg.server.host, "port": cfg.server.port},
        "registered_consumers": build_pipeline().consumers(),
        "hint": "use --serve to start the API, or --selftest to check the payload contract",
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
