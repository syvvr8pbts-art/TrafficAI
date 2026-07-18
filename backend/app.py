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
from typing import List

import numpy as np

# Root Detection dataclass, used only to build a synthetic payload in --selftest.
from detector import Detection

from .config import get_config
from .counter import CANONICAL_CLASSES, VehicleCounter
from .density import TrafficDensity
from .detector import FrameDetector, build_payload
from .logger import DetectionLogger
from .occupancy import LaneOccupancy
from .pipeline import AnalyticsPipeline
from .statistics import TrafficStatistics
from .utils import get_logger

logger = get_logger(__name__)


def build_pipeline() -> AnalyticsPipeline:
    """Create the analytics pipeline and register consumers in order.

    Order matters: each stage appends to the payload the next stage may read.
        Counter -> Density -> Occupancy -> Statistics -> Logger

    Adding a future module is a single ``.register(...)`` line here; the detector
    is never touched.
    """
    pipeline = AnalyticsPipeline()
    pipeline.register(VehicleCounter())
    pipeline.register(TrafficDensity())
    pipeline.register(LaneOccupancy())
    pipeline.register(TrafficStatistics())
    pipeline.register(DetectionLogger())
    return pipeline


# Maps canonical count keys -> the labels used in the terminal summary block.
_SUMMARY_LABELS = [
    ("car", "Cars"), ("bike", "Bikes"), ("auto-rickshaw", "Autos"),
    ("cycle", "Cycles"), ("bus", "Bus"), ("truck", "Truck"),
    ("ambulance", "Ambulance"),
]


def render_terminal_summary(payload: dict) -> str:
    """Format a clean per-frame summary block from a fully processed payload."""
    counts = payload.get("counts", {})
    lines = ["=" * 36, f"Frame: {payload.get('frame_id')}"]
    for key, label in _SUMMARY_LABELS:
        lines.append(f"{label:<14}: {counts.get(key, 0)}")
    lines.append(f"{'Total Vehicles':<14}: {payload.get('total', 0)}")
    lines.append(f"{'Traffic Density':<14}: {payload.get('density')}")
    lines.append(f"{'Lane Occupancy':<14}: {payload.get('occupancy')}%")
    lines.append("=" * 36)
    return "\n".join(lines)


def run_analysis(source: str, max_frames: int = 0) -> int:
    """Drive a video source through the full detection + analytics pipeline.

    Prints a per-frame terminal summary and logs each frame to CSV. The loop
    (not the detector) enriches the payload with the real frame dimensions so
    occupancy can be computed without changing detector.py.
    """
    import cv2

    cfg = get_config()
    detector = FrameDetector(cfg.detector)
    pipeline = build_pipeline()

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
            # Enrich payload with real frame size for the occupancy estimator.
            payload["frame_height"], payload["frame_width"] = frame.shape[0], frame.shape[1]
            pipeline.dispatch(payload)
            print(render_terminal_summary(payload))
            processed += 1
            if max_frames and processed >= max_frames:
                break
    finally:
        cap.release()
    logger.info("Analysis finished: %d frame(s) processed.", processed)
    return 0


def create_api():
    """Build the Flask app wired to a fresh pipeline and config."""
    from .api import create_app  # lazy: only import Flask path when serving
    return create_app(pipeline=build_pipeline(), config=get_config())


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
    parser.add_argument("--serve", action="store_true", help="Start the Flask API server")
    parser.add_argument("--selftest", action="store_true", help="Validate the payload contract (no model load)")
    parser.add_argument("--analyze", metavar="SOURCE", help="Run detection + analytics over a video source")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = no limit)")
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()

    if args.analyze:
        return run_analysis(args.analyze, max_frames=args.max_frames)

    if args.serve:
        cfg = get_config()
        app = create_api()
        logger.info("Serving API on http://%s:%d", cfg.server.host, cfg.server.port)
        app.run(host=cfg.server.host, port=cfg.server.port, debug=cfg.server.debug)
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
