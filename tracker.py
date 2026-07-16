"""
tracker.py

Multi-object tracking via ByteTrack (through the `supervision` library),
decoupled from the detector so each frame's raw detections become
persistent track IDs. This is what prevents duplicate counting and lets
vehicles survive brief occlusions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from config import TrackerConfig
from detector import Detection
from utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class TrackedObject:
    track_id: int
    xyxy: np.ndarray
    confidence: float
    class_id: int
    class_name: str

    @property
    def centroid(self) -> np.ndarray:
        x1, y1, x2, y2 = self.xyxy
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


class VehicleTracker:
    """Wraps supervision.ByteTrack to assign persistent IDs to detections."""

    def __init__(self, cfg: TrackerConfig) -> None:
        self.cfg = cfg
        self._tracker = self._build_tracker()

    def _build_tracker(self):
        try:
            import supervision as sv
        except ImportError as exc:
            raise ImportError(
                "supervision is required for ByteTrack. Install with: pip install supervision"
            ) from exc

        return sv.ByteTrack(
            track_activation_threshold=self.cfg.track_thresh,
            lost_track_buffer=self.cfg.track_buffer,
            minimum_matching_threshold=self.cfg.match_thresh,
            frame_rate=self.cfg.frame_rate,
        )

    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        import supervision as sv

        if not detections:
            sv_detections = sv.Detections.empty()
        else:
            xyxy = np.array([d.xyxy for d in detections], dtype=np.float32)
            confidence = np.array([d.confidence for d in detections], dtype=np.float32)
            class_id = np.array([d.class_id for d in detections], dtype=int)
            sv_detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

        tracked = self._tracker.update_with_detections(sv_detections)

        class_lookup = {d.class_id: d.class_name for d in detections}
        results: List[TrackedObject] = []
        if tracked.tracker_id is None:
            return results

        for xyxy, conf, cls_id, track_id in zip(
            tracked.xyxy, tracked.confidence, tracked.class_id, tracked.tracker_id
        ):
            results.append(
                TrackedObject(
                    track_id=int(track_id),
                    xyxy=xyxy.astype(np.float32),
                    confidence=float(conf) if conf is not None else 0.0,
                    class_id=int(cls_id),
                    class_name=class_lookup.get(int(cls_id), f"class_{cls_id}"),
                )
            )
        return results
