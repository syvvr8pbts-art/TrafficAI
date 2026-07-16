"""
lane_detection.py

User-defined lane polygons and assignment of tracked vehicles to lanes.
Lanes are described in a JSON file (see configs/lanes_example.json) as a
list of named polygons in image pixel coordinates. Use tools/define_lanes.py
to draw these interactively on a real frame from your own footage instead
of hand-editing coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from shapely.geometry import Point, Polygon

from utils import load_json, setup_logger

logger = setup_logger(__name__)


@dataclass
class Lane:
    name: str
    polygon: Polygon
    points: List[List[int]]
    real_world_area_sqm: Optional[float] = None  # optional, feeds density calc


class LaneManager:
    """Loads lane polygons from a JSON file and assigns points/tracks to lanes."""

    def __init__(self, lanes_config_path: str) -> None:
        self.lanes: List[Lane] = self._load_lanes(lanes_config_path)

    def _load_lanes(self, path: str) -> List[Lane]:
        data = load_json(path)
        lanes: List[Lane] = []
        for lane_def in data.get("lanes", []):
            points = lane_def["points"]
            lanes.append(
                Lane(
                    name=lane_def["name"],
                    polygon=Polygon(points),
                    points=points,
                    real_world_area_sqm=lane_def.get("real_world_area_sqm"),
                )
            )
        logger.info("Loaded %d lane(s) from %s", len(lanes), path)
        return lanes

    def assign_lane(self, centroid: np.ndarray) -> Optional[str]:
        """Return the name of the lane containing this point, or None if it's in no lane."""
        point = Point(float(centroid[0]), float(centroid[1]))
        for lane in self.lanes:
            if lane.polygon.contains(point):
                return lane.name
        return None

    def get_lane_names(self) -> List[str]:
        return [lane.name for lane in self.lanes]

    def get_lane(self, name: str) -> Optional[Lane]:
        for lane in self.lanes:
            if lane.name == name:
                return lane
        return None
