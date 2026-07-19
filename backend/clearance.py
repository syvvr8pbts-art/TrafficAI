"""
backend/clearance.py

Intersection clearance (signal-intelligence stage 3). Single responsibility:
decide whether the central intersection is physically clear of vehicles, using
the existing detections rather than a blind timer.

A configurable rectangular ROI represents the intersection centre. If ANY
detection bounding box overlaps the ROI, the intersection is OCCUPIED; otherwise
CLEAR. When wired to the signal controller, it reports this each frame so the
controller holds in ALL_RED/CLEARANCE until the box is actually clear (occupancy
info preferred over a fixed timer).

ROI is given in normalized [0,1] coordinates by default and resolved to pixels
using the frame size the app loop places on the payload (falling back to a
configured default) -- so the detector never has to change.

Appends to the payload:
    payload["intersection"] = {"status": "CLEAR" | "OCCUPIED", "roi": [x1,y1,x2,y2]}
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .utils import get_logger

logger = get_logger(__name__)

# Central rectangle as fractions of the frame (x1, y1, x2, y2).
DEFAULT_ROI: Tuple[float, float, float, float] = (0.3, 0.3, 0.7, 0.7)
DEFAULT_FRAME_SIZE: Tuple[int, int] = (1920, 1080)


class IntersectionClearance:
    """Reports whether the intersection ROI is occupied by any vehicle."""

    name = "clearance"

    def __init__(
        self,
        controller: Optional[Any] = None,
        roi: Tuple[float, float, float, float] = DEFAULT_ROI,
        normalized: bool = True,
        default_size: Tuple[int, int] = DEFAULT_FRAME_SIZE,
    ) -> None:
        self.controller = controller
        self.roi = roi
        self.normalized = normalized
        self.default_size = default_size

    def _roi_pixels(self, payload: Dict[str, Any]) -> Tuple[float, float, float, float]:
        """Resolve the ROI to pixel coordinates for the current frame."""
        if not self.normalized:
            return self.roi
        w = payload.get("frame_width") or self.default_size[0]
        h = payload.get("frame_height") or self.default_size[1]
        x1, y1, x2, y2 = self.roi
        return (x1 * w, y1 * h, x2 * w, y2 * h)

    @staticmethod
    def _overlaps(bbox: List[float], roi: Tuple[float, float, float, float]) -> bool:
        """Axis-aligned rectangle overlap test."""
        if not bbox or len(bbox) != 4:
            return False
        x1, y1, x2, y2 = bbox
        rx1, ry1, rx2, ry2 = roi
        return not (x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2)

    def is_occupied(self, detections: List[Dict[str, Any]], roi: Tuple[float, float, float, float]) -> bool:
        return any(self._overlaps(d.get("bbox"), roi) for d in detections)

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Consumer hook: append intersection status and notify the controller."""
        roi = self._roi_pixels(payload)
        occupied = self.is_occupied(payload.get("detections", []), roi)
        payload["intersection"] = {
            "status": "OCCUPIED" if occupied else "CLEAR",
            "roi": [round(v, 1) for v in roi],
        }
        if self.controller is not None:
            self.controller.report_clearance(occupied)
        return payload
