"""
backend/occupancy.py

Lane/road occupancy estimator (analytics stage 3). Single responsibility:
approximate how much of the frame is covered by vehicles.

Estimation method (documented):
    occupancy% = ( sum of detected bounding-box pixel areas / frame pixel area ) * 100

    * Each detection contributes width*height of its [x1,y1,x2,y2] box.
    * Overlapping boxes are NOT de-duplicated, so this is a fast, upper-leaning
      approximation of coverage rather than an exact free-space measurement.
    * The result is clamped to the 0-100% range.

Frame area is taken from ``frame_width``/``frame_height`` in the payload when the
caller provides them (the app loop sets them from the real frame). If absent, a
configured frame size is used as a fallback -- so this module never needs the
detector to change.

Appends to the payload:
    payload["occupancy"] = <float 0..100, one decimal>
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .utils import get_logger

logger = get_logger(__name__)

DEFAULT_FRAME_SIZE: Tuple[int, int] = (1920, 1080)  # fallback (w, h)


class LaneOccupancy:
    """Estimates percentage of frame area covered by detected vehicles."""

    name = "occupancy"

    def __init__(
        self,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        default_size: Tuple[int, int] = DEFAULT_FRAME_SIZE,
    ) -> None:
        self._w = frame_width
        self._h = frame_height
        self._default = default_size

    def set_frame_size(self, width: int, height: int) -> None:
        """Optionally fix the frame size (e.g. once, from the video stream)."""
        self._w, self._h = width, height

    def _frame_area(self, payload: Dict[str, Any]) -> float:
        """Resolve frame area from payload dims, configured dims, then default."""
        width = payload.get("frame_width") or self._w or self._default[0]
        height = payload.get("frame_height") or self._h or self._default[1]
        return float(max(width * height, 1))

    def estimate(self, detections: List[Dict[str, Any]], frame_area: float) -> float:
        """Return occupancy percentage (0..100) for the given detections."""
        covered = 0.0
        for det in detections:
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            covered += max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return round(min(covered / frame_area * 100.0, 100.0), 1)

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Consumer hook: append the occupancy percentage."""
        payload["occupancy"] = self.estimate(payload.get("detections", []), self._frame_area(payload))
        return payload
