"""
backend/detector.py

Backend detection adapter. This is the ONLY backend module that touches the YOLO
model, and it does so by wrapping the existing, unchanged project-root
``VehicleDetector`` -- the detection logic, weights, class filter, and bounding
boxes behave exactly as before. Its single responsibility is to turn a raw frame
into a clean, JSON-serializable *detection payload*:

    {
        "frame_id": 1,
        "timestamp": 1700000000.0,
        "detections": [
            {"class": "car", "confidence": 0.93, "bbox": [x1, y1, x2, y2]},
            ...
        ]
    }

Downstream analytics modules (counter, density, occupancy, recommendation,
ambulance, statistics, logger, snapshot) consume this dict and NEVER import the
model -- so they can be added without ever touching this file.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

# Existing, unchanged detection pipeline (absolute imports -> top-level modules).
from config import DetectorConfig
from detector import Detection, VehicleDetector

from .config import get_config
from .utils import epoch_now, get_logger, to_float_list

logger = get_logger(__name__)


def detection_to_dict(detection: Detection) -> Dict[str, Any]:
    """Convert one root ``Detection`` dataclass into a serializable dict.

    Values pass through untouched (same class name, confidence, and pixel bbox).
    """
    return {
        "class": detection.class_name,
        "confidence": float(detection.confidence),
        "bbox": to_float_list(detection.xyxy),  # [x1, y1, x2, y2]
    }


def build_payload(
    frame_id: int,
    detections: List[Detection],
    timestamp: Optional[float] = None,
) -> Dict[str, Any]:
    """Assemble the structured per-frame payload from raw detections.

    Pure function (no model involved) so it is trivially unit-testable.
    """
    return {
        "frame_id": frame_id,
        "timestamp": epoch_now() if timestamp is None else timestamp,
        "detections": [detection_to_dict(d) for d in detections],
    }


class FrameDetector:
    """Frame -> structured detection payload.

    Thin, stateful wrapper around the root ``VehicleDetector``. It owns an
    auto-incrementing frame counter so callers may omit ``frame_id``, but the
    actual inference is delegated verbatim to the existing detector.
    """

    def __init__(self, cfg: Optional[DetectorConfig] = None) -> None:
        # Fall back to the root detector configuration when none is supplied.
        self._detector = VehicleDetector(cfg or get_config().detector)
        self._frame_counter = 0

    @property
    def device(self) -> str:
        """The concrete device resolved by the underlying detector."""
        return self._detector.device

    def process(
        self,
        frame: np.ndarray,
        frame_id: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run detection on a single BGR frame and return its payload dict.

        The detection call itself is unchanged; only the output is repackaged.
        """
        if frame_id is None:
            self._frame_counter += 1
            frame_id = self._frame_counter
        detections = self._detector.detect(frame)  # identical to the legacy path
        return build_payload(frame_id, detections, timestamp)
