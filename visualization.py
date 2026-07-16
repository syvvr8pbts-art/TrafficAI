"""
visualization.py

Draws bounding boxes, track IDs, lane polygons, and a small analytics HUD
onto each frame. Kept separate from analysis so the same TrafficAnalyzer
output could later feed a web dashboard instead of (or alongside) this
OpenCV overlay.
"""
from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np

from lane_detection import LaneManager
from tracker import TrackedObject

_LEVEL_COLORS = {
    "green": (0, 200, 0),
    "yellow": (0, 220, 220),
    "orange": (0, 140, 255),
    "red": (0, 0, 255),
}

_BOX_COLOR = (60, 200, 60)
_TEXT_COLOR = (255, 255, 255)


def draw_lanes(frame: np.ndarray, lane_manager: LaneManager) -> np.ndarray:
    overlay = frame.copy()
    for lane in lane_manager.lanes:
        pts = np.array(lane.points, dtype=np.int32)
        cv2.polylines(overlay, [pts], isClosed=True, color=(255, 200, 0), thickness=2)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(overlay, lane.name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
    return cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)


def draw_tracks(frame: np.ndarray, tracked_objects: List[TrackedObject]) -> np.ndarray:
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj.xyxy.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), _BOX_COLOR, 2)
        label = f"#{obj.track_id} {obj.class_name} {obj.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), _BOX_COLOR, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TEXT_COLOR, 1)
    return frame


def draw_hud(frame: np.ndarray, summary: Dict[str, object], fps: float) -> np.ndarray:
    """Stacks a small stats panel above the annotated frame."""
    h, w = frame.shape[:2]
    panel_h = 90
    panel = np.zeros((panel_h, w, 3), dtype=np.uint8)

    total = summary.get("total_unique_vehicles", 0)
    by_type = summary.get("counts_by_type", {})
    type_str = " | ".join(f"{k}:{v}" for k, v in by_type.items()) or "no vehicles yet"

    cv2.putText(panel, f"FPS: {fps:.1f}   Total vehicles: {total}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, _TEXT_COLOR, 1)
    cv2.putText(panel, type_str, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _TEXT_COLOR, 1)

    x = 10
    for lane_name, cong in summary.get("congestion_by_lane", {}).items():
        color = _LEVEL_COLORS.get(cong["level"], (200, 200, 200))
        text = f"{lane_name}: {cong['level'].upper()} ({cong['occupancy']})"
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(panel, (x, 60), (x + tw + 10, 82), color, -1)
        cv2.putText(panel, text, (x + 5, 77), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        x += tw + 20

    return np.vstack([panel, frame])
