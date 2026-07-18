"""
backend/statistics.py

Cumulative traffic statistics (analytics stage 4). Single responsibility:
accumulate running totals across every processed frame for the whole session.

It reads the per-frame ``counts`` produced by the counter and maintains totals
per class plus frames processed and total vehicles detected. State is held in
the instance, so one TrafficStatistics object spans the run.

Appends to the payload:
    payload["statistics"] = {
        "frames": .., "vehicles": ..,
        "cars": .., "bikes": .., "cycles": .., "autos": ..,
        "buses": .., "trucks": .., "ambulances": ..,
    }
"""
from __future__ import annotations

from typing import Any, Dict

from .counter import CANONICAL_CLASSES
from .utils import get_logger

logger = get_logger(__name__)

# Map canonical per-frame count keys -> cumulative statistic keys (pluralized).
STAT_KEYS: Dict[str, str] = {
    "car": "cars",
    "bike": "bikes",
    "cycle": "cycles",
    "auto-rickshaw": "autos",
    "bus": "buses",
    "truck": "trucks",
    "ambulance": "ambulances",
}


class TrafficStatistics:
    """Accumulates session-long totals from per-frame counts."""

    name = "statistics"

    def __init__(self) -> None:
        self._frames = 0
        self._totals: Dict[str, int] = {STAT_KEYS[c]: 0 for c in CANONICAL_CLASSES}

    def update(self, counts: Dict[str, int]) -> Dict[str, Any]:
        """Fold one frame's counts into the running totals; return a snapshot."""
        self._frames += 1
        for canonical, stat_key in STAT_KEYS.items():
            self._totals[stat_key] += int(counts.get(canonical, 0))
        return self.snapshot()

    def snapshot(self) -> Dict[str, Any]:
        """Return the current cumulative statistics as a plain dict."""
        return {
            "frames": self._frames,
            "vehicles": sum(self._totals.values()),
            **self._totals,
        }

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Consumer hook: update cumulative stats and append the snapshot."""
        payload["statistics"] = self.update(payload.get("counts", {}))
        return payload
