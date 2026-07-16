"""
system_snapshot.py

The canonical, immutable data model for the whole Smart Traffic Management
System. Every producing module (analyzer, decision engine, signal controller,
and -- in later milestones -- green corridor and signal network) contributes to
a SystemSnapshot, and every consumer (HUD, Flask dashboard) reads ONLY from it.
No module should query another module's internals directly.

The structures reserve `emergency` and `network` fields now so Milestones 3-4
populate them instead of introducing parallel data structures later.

StateManager is the single source of truth at runtime: producers `publish()` a
fresh snapshot each processed frame; consumers (possibly on other threads, e.g.
the dashboard) read the `latest()` one. Snapshots are frozen dataclasses built
fresh each frame, so readers never observe a half-updated state.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class IntersectionSnapshot:
    """Immutable per-intersection view for one processed frame."""
    intersection_id: str
    display_name: str
    analytics: Dict[str, object]           # subset of TrafficAnalyzer.summary()
    decision: Dict[str, object]            # TrafficDecision.to_dict()
    signal: Dict[str, object]              # SignalState.to_dict()
    emergency: Optional[Dict[str, object]] = None  # populated in Milestone 3
    network: Optional[Dict[str, object]] = None    # populated in Milestone 4

    def to_dict(self) -> Dict[str, object]:
        return {
            "intersection_id": self.intersection_id,
            "display_name": self.display_name,
            "analytics": self.analytics,
            "decision": self.decision,
            "signal": self.signal,
            "emergency": self.emergency,
            "network": self.network,
        }


@dataclass(frozen=True)
class SystemSnapshot:
    """Immutable whole-system view for one processed frame."""
    timestamp: str
    frame_index: int
    fps: float
    total_unique_vehicles: int
    intersections: Tuple[IntersectionSnapshot, ...] = ()

    @classmethod
    def create(
        cls,
        frame_index: int,
        fps: float,
        total_unique_vehicles: int,
        intersections: List[IntersectionSnapshot],
        timestamp: Optional[str] = None,
    ) -> "SystemSnapshot":
        return cls(
            timestamp=timestamp or datetime.now().isoformat(),
            frame_index=frame_index,
            fps=round(fps, 1),
            total_unique_vehicles=total_unique_vehicles,
            intersections=tuple(intersections),
        )

    def intersection(self, intersection_id: str) -> Optional[IntersectionSnapshot]:
        for inter in self.intersections:
            if inter.intersection_id == intersection_id:
                return inter
        return None

    def to_dict(self) -> Dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "fps": self.fps,
            "total_unique_vehicles": self.total_unique_vehicles,
            "intersections": [inter.to_dict() for inter in self.intersections],
        }


class StateManager:
    """Thread-safe holder of the latest SystemSnapshot (single source of truth)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: Optional[SystemSnapshot] = None

    def publish(self, snapshot: SystemSnapshot) -> None:
        with self._lock:
            self._latest = snapshot

    def latest(self) -> Optional[SystemSnapshot]:
        with self._lock:
            return self._latest
