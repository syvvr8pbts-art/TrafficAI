"""
backend/density.py

Traffic density classifier (analytics stage 2). Single responsibility: map the
total vehicle count for a frame to a coarse density label.

Thresholds are configurable (defaults per spec):
    0 .. low_max      -> LOW      (default low_max = 10)
    low_max+1 .. med_max -> MEDIUM (default med_max = 25)
    med_max+1 ..      -> HIGH

Appends to the payload:
    payload["density"] = "LOW" | "MEDIUM" | "HIGH"
"""
from __future__ import annotations

from typing import Any, Dict

from .utils import get_logger

logger = get_logger(__name__)

DEFAULT_LOW_MAX = 10    # inclusive upper bound for LOW
DEFAULT_MEDIUM_MAX = 25  # inclusive upper bound for MEDIUM


class TrafficDensity:
    """Classifies traffic density from a total vehicle count."""

    name = "density"

    def __init__(self, low_max: int = DEFAULT_LOW_MAX, medium_max: int = DEFAULT_MEDIUM_MAX) -> None:
        self.low_max = low_max
        self.medium_max = medium_max

    def classify(self, total: int) -> str:
        """Return the density label for a given total vehicle count."""
        if total <= self.low_max:
            return "LOW"
        if total <= self.medium_max:
            return "MEDIUM"
        return "HIGH"

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Consumer hook: read the total (from the counter) and append density."""
        total = payload.get("total")
        if total is None:  # fall back if run standalone without the counter
            total = sum(payload.get("counts", {}).values())
        payload["density"] = self.classify(int(total))
        return payload
