"""
decision_engine.py

AI decision engine for traffic prioritization. It delegates per-lane demand
scoring to an interchangeable DecisionPolicy (WeightedDemandPolicy by default)
and is itself responsible only for ranking those demands into a service order
and naming the active lane -- scoped to a single Intersection.

It answers "who needs the green next and in what order", NOT "for how long":
converting priorities into green/yellow/red durations and countdowns is the
signal controller's job. The engine reads only the analytics summary dict, so
it stays decoupled from the detector, tracker, and OpenCV.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import DecisionConfig
from decision_policy import DecisionPolicy, LaneDemand, WeightedDemandPolicy
from intersection import Intersection
from utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class TrafficDecision:
    """Priority output for one intersection, consumed by the signal controller.

    Note: intentionally contains NO signal timings -- only demand and ordering.
    """
    intersection_id: str
    demands: Dict[str, LaneDemand] = field(default_factory=dict)
    service_order: List[str] = field(default_factory=list)  # lanes, most→least demand
    active_lane: Optional[str] = None  # first lane in service_order

    def to_dict(self) -> Dict[str, object]:
        """JSON-serializable view for snapshots / the dashboard."""
        return {
            "intersection_id": self.intersection_id,
            "active_lane": self.active_lane,
            "service_order": self.service_order,
            "demands": {
                lane: {
                    "occupancy": d.occupancy,
                    "total_count": d.total_count,
                    "congestion_level": d.congestion_level,
                    "ratio": round(d.ratio, 3),
                    "demand_score": round(d.demand_score, 3),
                    "components": {k: round(v, 3) for k, v in d.components.items()},
                }
                for lane, d in self.demands.items()
            },
        }
 
     
class DecisionEngine:
    """Ranks an intersection's lanes by demand using a pluggable policy.

    Feed it an Intersection and that intersection's TrafficAnalyzer.summary();
    it returns a TrafficDecision. The demand math lives entirely in the policy.
    """

    def __init__(self, cfg: DecisionConfig, policy: Optional[DecisionPolicy] = None) -> None:
        self.cfg = cfg
        self.policy: DecisionPolicy = policy or WeightedDemandPolicy(cfg)

    def decide(self, intersection: Intersection, summary: Dict[str, object]) -> TrafficDecision:
        """Rank the intersection's lanes by blended demand (via the policy)."""
        demands = self.policy.evaluate(intersection, summary)

        service_order = sorted(
            demands, key=lambda ln: demands[ln].demand_score, reverse=True
        )
        active_lane = service_order[0] if service_order else None

        logger.debug(
            "Decision[%s]: policy=%s active=%s order=%s",
            intersection.intersection_id,
            type(self.policy).__name__,
            active_lane,
            service_order,
        )
        return TrafficDecision(
            intersection_id=intersection.intersection_id,
            demands=demands,
            service_order=service_order,
            active_lane=active_lane,
        )
