"""
decision_policy.py

Pluggable demand-evaluation policies for the decision engine. A DecisionPolicy
maps an intersection's analytics summary to a per-lane LaneDemand. Swapping the
policy changes *how* demand is scored without touching the engine that ranks
and orders lanes.

WeightedDemandPolicy is the default: a weighted blend of occupancy ratio,
normalized cumulative count, congestion severity, and (optionally) vehicle-type
intensity. Alternative policies (queue-length, wait-time fairness, learned) can
implement DecisionPolicy and be injected into DecisionEngine unchanged.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional

from config import DecisionConfig
from intersection import Intersection


@dataclass
class LaneDemand:
    """A lane's blended demand for green time this decision cycle."""
    lane: str
    occupancy: int
    total_count: int
    congestion_level: str
    ratio: float          # occupancy ratio in [0, 1] from TrafficAnalyzer
    demand_score: float   # blended demand in [0, 1]
    components: Dict[str, float] = field(default_factory=dict)  # per-metric breakdown


class DecisionPolicy(ABC):
    """Strategy interface: evaluate per-lane demand for one intersection."""

    @abstractmethod
    def evaluate(
        self, intersection: Intersection, summary: Dict[str, object]
    ) -> Dict[str, LaneDemand]:
        """Return {lane_name: LaneDemand} covering every lane on the
        intersection (lanes absent from the summary get zero demand).
        """
        raise NotImplementedError


class WeightedDemandPolicy(DecisionPolicy):
    """Default policy: weighted blend of density, count, congestion, and
    (optional) vehicle-type intensity. All weights come from DecisionConfig.
    """

    def __init__(self, cfg: DecisionConfig) -> None:
        self.cfg = cfg
        self._max_type_weight = max(cfg.vehicle_type_weights.values(), default=1.0)

    def _congestion_severity(self, level: str) -> float:
        return float(self.cfg.congestion_severity.get(level, 0.0))

    def _type_intensity(self, lane_type_counts: Dict[str, int]) -> Optional[float]:
        """Average per-vehicle type weight for a lane, normalized to [0, 1].

        Returns None when no per-lane type counts are available so the term can
        be dropped from the blend rather than counted as zero.
        """
        total = sum(lane_type_counts.values())
        if total <= 0:
            return None
        weighted = sum(
            self.cfg.vehicle_type_weights.get(vtype, 1.0) * n
            for vtype, n in lane_type_counts.items()
        )
        avg_weight = weighted / total
        return min(avg_weight / self._max_type_weight, 1.0)

    def _build_demand(
        self,
        lane: str,
        cong: Dict[str, object],
        total_count: int,
        count_norm: float,
        lane_type_counts: Optional[Dict[str, int]],
    ) -> LaneDemand:
        ratio = float(cong.get("ratio", 0.0))
        occupancy = int(cong.get("occupancy", 0))
        level = str(cong.get("level", "green"))
        severity = self._congestion_severity(level)

        # Only include the type term when we have data for it, then renormalize
        # over whichever terms are present.
        terms: Dict[str, tuple] = {
            "occupancy": (self.cfg.occupancy_weight, ratio),
            "count": (self.cfg.count_weight, count_norm),
            "congestion": (self.cfg.congestion_weight, severity),
        }
        if lane_type_counts is not None:
            type_intensity = self._type_intensity(lane_type_counts)
            if type_intensity is not None:
                terms["type"] = (self.cfg.type_weight, type_intensity)

        weight_sum = sum(w for w, _ in terms.values()) or 1.0
        components = {name: (w * v) / weight_sum for name, (w, v) in terms.items()}
        demand_score = sum(components.values())

        return LaneDemand(
            lane=lane,
            occupancy=occupancy,
            total_count=total_count,
            congestion_level=level,
            ratio=ratio,
            demand_score=demand_score,
            components=components,
        )

    def evaluate(
        self, intersection: Intersection, summary: Dict[str, object]
    ) -> Dict[str, LaneDemand]:
        congestion_by_lane: Dict[str, dict] = summary.get("congestion_by_lane", {})  # type: ignore[assignment]
        counts_by_lane: Dict[str, int] = summary.get("counts_by_lane", {})  # type: ignore[assignment]
        # Optional per-lane, per-type counts: {lane: {vtype: n}}. Absent by
        # default; when present it activates the vehicle-type demand term.
        counts_by_lane_type: Dict[str, Dict[str, int]] = summary.get("counts_by_lane_type", {})  # type: ignore[assignment]

        max_count = max(counts_by_lane.values(), default=0)

        demands: Dict[str, LaneDemand] = {}
        for lane in intersection.lane_names:
            cong = congestion_by_lane.get(lane, {})
            total_count = int(counts_by_lane.get(lane, 0))
            count_norm = (total_count / max_count) if max_count > 0 else 0.0
            lane_type_counts = counts_by_lane_type.get(lane)
            demands[lane] = self._build_demand(
                lane, cong, total_count, count_norm, lane_type_counts
            )
        return demands
