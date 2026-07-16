"""
traffic_analysis.py

Turns per-frame tracked + lane-assigned vehicles into traffic analytics:
unique counts (total, per-lane, per-type), per-minute counts, and a simple
occupancy-based congestion score with a green/yellow/orange/red rating.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from config import AnalysisConfig
from lane_detection import LaneManager
from tracker import TrackedObject
from utils import ensure_dir, setup_logger

logger = setup_logger(__name__)


@dataclass
class LaneStats:
    total_count: int = 0
    counts_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    current_occupancy: int = 0  # vehicles inside the lane polygon this frame


class TrafficAnalyzer:
    """Stateful analyzer: feed it tracked + lane-tagged detections every frame."""

    def __init__(self, cfg: AnalysisConfig, lane_manager: LaneManager) -> None:
        self.cfg = cfg
        self.lane_manager = lane_manager
        self._seen_track_ids: Set[int] = set()
        self._lane_stats: Dict[str, LaneStats] = {
            name: LaneStats() for name in lane_manager.get_lane_names()
        }
        self._type_totals: Dict[str, int] = defaultdict(int)
        self._minute_bucket: Dict[str, int] = defaultdict(int)  # "YYYY-mm-dd HH:MM" -> count
        self._start_time = datetime.now()

    def update(self, tracked_objects: List[TrackedObject], lane_of: Dict[int, Optional[str]]) -> None:
        """Call once per processed frame with this frame's tracked objects and
        a {track_id: lane_name_or_None} mapping from LaneManager.assign_lane.
        """
        occupancy_this_frame: Dict[str, int] = defaultdict(int)

        for obj in tracked_objects:
            lane_name = lane_of.get(obj.track_id)
            if lane_name is not None:
                occupancy_this_frame[lane_name] += 1

            if obj.track_id in self._seen_track_ids:
                continue  # already counted once -- prevents duplicate counting
            self._seen_track_ids.add(obj.track_id)

            self._type_totals[obj.class_name] += 1
            minute_key = datetime.now().strftime("%Y-%m-%d %H:%M")
            self._minute_bucket[minute_key] += 1

            if lane_name is not None:
                stats = self._lane_stats[lane_name]
                stats.total_count += 1
                stats.counts_by_type[obj.class_name] += 1

        for lane_name, stats in self._lane_stats.items():
            stats.current_occupancy = occupancy_this_frame.get(lane_name, 0)

    def congestion_for_lane(self, lane_name: str) -> Dict[str, object]:
        """Occupancy-ratio-based congestion score and green/yellow/orange/red
        label. ratio = vehicles currently in the lane / a rough capacity
        estimate derived from the lane's real-world area (if provided) or a
        flat default, assuming ~10 sqm per vehicle including spacing.
        """
        lane = self.lane_manager.get_lane(lane_name)
        stats = self._lane_stats.get(lane_name, LaneStats())
        area = (
            lane.real_world_area_sqm
            if lane and lane.real_world_area_sqm
            else self.cfg.density_area_sqm_per_lane
        )
        estimated_capacity = max(area / 10.0, 1.0)
        ratio = min(stats.current_occupancy / estimated_capacity, 1.0)

        thresholds = self.cfg.congestion_thresholds
        if ratio <= thresholds["green"]:
            level = "green"
        elif ratio <= thresholds["yellow"]:
            level = "yellow"
        elif ratio <= thresholds["orange"]:
            level = "orange"
        else:
            level = "red"

        return {"lane": lane_name, "occupancy": stats.current_occupancy, "ratio": round(ratio, 2), "level": level}

    def summary(self) -> Dict[str, object]:
        return {
            "session_start": self._start_time.isoformat(),
            "elapsed_seconds": (datetime.now() - self._start_time).total_seconds(),
            "total_unique_vehicles": len(self._seen_track_ids),
            "counts_by_type": dict(self._type_totals),
            "counts_by_lane": {name: stats.total_count for name, stats in self._lane_stats.items()},
            "congestion_by_lane": {name: self.congestion_for_lane(name) for name in self._lane_stats},
            "vehicles_per_minute": dict(self._minute_bucket),
        }

    def export_csv(self, output_dir: str) -> Path:
        out_dir = ensure_dir(output_dir)
        path = out_dir / f"traffic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["total_unique_vehicles", len(self._seen_track_ids)])
            for vtype, count in self._type_totals.items():
                writer.writerow([f"count_{vtype}", count])
            for lane_name, stats in self._lane_stats.items():
                writer.writerow([f"count_lane_{lane_name}", stats.total_count])
                cong = self.congestion_for_lane(lane_name)
                writer.writerow([f"congestion_lane_{lane_name}", cong["level"]])
        logger.info("Wrote traffic report to %s", path)
        return path
