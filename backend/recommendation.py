"""
backend/recommendation.py

Signal-timing recommendation engine (signal-intelligence stage 1). Single
responsibility: recommend a green-light duration for the current frame based on
traffic conditions. It reads the analytics payload (density, counts, occupancy)
and outputs a recommended green time in seconds.

It is intentionally *stateless* and *signal-agnostic*: it knows nothing about
lanes, phases, or the controller's current state -- it only answers "given this
much traffic, how long should a green be?". The controller decides how/when to
use that number.

Default mapping (configurable):
    LOW    -> 10 seconds
    MEDIUM -> 20 seconds
    HIGH   -> 30 seconds

Appends to the payload:
    payload["recommended_green"] = <int seconds>
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .utils import get_logger

logger = get_logger(__name__)

# Density label -> recommended green seconds.
DEFAULT_GREEN_BY_DENSITY: Dict[str, int] = {
    "LOW": 10,
    "MEDIUM": 20,
    "HIGH": 30,
}


class RecommendationEngine:
    """Recommends a green duration from traffic density."""

    name = "recommendation"

    def __init__(
        self,
        green_by_density: Optional[Dict[str, int]] = None,
        default_green: int = 15,
    ) -> None:
        self.green_by_density = {
            k.upper(): int(v)
            for k, v in (green_by_density or DEFAULT_GREEN_BY_DENSITY).items()
        }
        self.default_green = int(default_green)

    def recommend(self, density: Optional[str]) -> int:
        """Return the recommended green seconds for a density label."""
        return self.green_by_density.get((density or "").upper(), self.default_green)

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Consumer hook: append the recommended green time."""
        payload["recommended_green"] = self.recommend(payload.get("density"))
        return payload
