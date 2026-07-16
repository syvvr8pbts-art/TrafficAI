"""
intersection.py

The Intersection abstraction -- the unit the whole control system operates on.
Even the single-camera hackathon demo runs as exactly one Intersection, so
scaling to several later (each with its own lanes, analyzer, decision engine,
and signal controller) needs no structural change: you just register more.

This module holds *identity and topology* only (id, the lanes it governs, and
which downstream intersections it can notify). Live analytics and control
state live in the per-frame snapshot, not here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Intersection:
    """Identity + topology for one signalized intersection."""
    intersection_id: str
    lane_names: List[str]
    display_name: str = ""
    # Downstream intersections this one can warn when an ambulance is
    # approaching (drives the signal-to-signal green-corridor propagation).
    downstream_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.display_name:
            self.display_name = self.intersection_id
        # De-duplicate lane names while preserving their configured order.
        seen: set[str] = set()
        ordered: List[str] = []
        for name in self.lane_names:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        self.lane_names = ordered

    def has_lane(self, lane_name: str) -> bool:
        return lane_name in self.lane_names
