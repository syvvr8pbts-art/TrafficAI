"""
backend/counter.py

Vehicle counter (analytics stage 1). Single responsibility: given a frame's
detection payload, count detections per canonical vehicle class and the total.

Detection class names coming from the model may vary (e.g. the COCO checkpoint
emits "motorcycle"/"bicycle"); this module normalizes them to the seven
canonical output classes via a configurable alias map. This normalization lives
ONLY here, so no other analytics module needs to know about model naming.

Appends to the payload:
    payload["counts"] = {"car": .., "bike": .., ... "ambulance": ..}
    payload["total"]  = <sum of counts>
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .utils import get_logger

logger = get_logger(__name__)

# Canonical output classes, in display order.
CANONICAL_CLASSES: List[str] = [
    "car", "bike", "cycle", "auto-rickshaw", "bus", "truck", "ambulance",
]

# Maps raw detector labels (lower-cased) -> canonical class. Extend as needed
# for custom-trained models; unknown labels are simply ignored.
DEFAULT_ALIASES: Dict[str, str] = {
    "car": "car",
    "bike": "bike", "motorbike": "bike", "motorcycle": "bike",
    "cycle": "cycle", "bicycle": "cycle",
    "auto-rickshaw": "auto-rickshaw", "auto_rickshaw": "auto-rickshaw",
    "auto": "auto-rickshaw", "rickshaw": "auto-rickshaw",
    "three wheelers -cng-": "auto-rickshaw",
    "bus": "bus", "minibus": "bus",
    "truck": "truck",
    "ambulance": "ambulance",
}


class VehicleCounter:
    """Counts detections per canonical vehicle class for a single frame."""

    name = "counter"

    def __init__(
        self,
        aliases: Optional[Dict[str, str]] = None,
        classes: Optional[List[str]] = None,
    ) -> None:
        self.classes = list(classes or CANONICAL_CLASSES)
        self.aliases = {k.lower(): v for k, v in (aliases or DEFAULT_ALIASES).items()}

    def count(self, detections: List[Dict[str, Any]]) -> Dict[str, int]:
        """Return a {canonical_class: count} dict (all classes present, zero-filled)."""
        counts = {c: 0 for c in self.classes}
        for det in detections:
            raw = str(det.get("class", "")).strip().lower()
            canonical = self.aliases.get(raw)
            if canonical in counts:
                counts[canonical] += 1
        return counts

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Consumer hook: append per-class counts and the total to the payload."""
        counts = self.count(payload.get("detections", []))
        payload["counts"] = counts
        payload["total"] = sum(counts.values())
        return payload
