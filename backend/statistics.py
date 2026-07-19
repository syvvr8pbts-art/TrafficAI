"""
backend/statistics.py

Cumulative traffic statistics (analytics stage 4). Single responsibility:
accumulate TRUE unique vehicle totals for the whole session -- i.e. how many
distinct vehicles have passed through the monitored area, NOT how many boxes
were seen per frame.

How uniqueness is achieved:
  * The existing ByteTrack tracker (root ``tracker.VehicleTracker``) is reused to
    assign a persistent ``track_id`` to every detection across frames.
  * A vehicle is counted exactly ONCE, as soon as its track has been continuously
    tracked for the confirmation window (>= confirm_frames, default 0.3s). No
    counting-line crossing is required, so slow-moving / congested traffic that
    never crosses a line is still counted. (The line helpers are retained but no
    longer gate cumulative counting.)
  * A ``counted_ids`` set per class guarantees a given ``track_id`` is never
    counted twice, so the totals only ever increase.

This is a drop-in replacement: the consumer interface (``process`` appending
``payload["statistics"]``) and all output keys are unchanged -- only their
semantics change from "sum of per-frame counts" to "unique vehicles counted".
The per-frame ``counts`` produced by counter.py (the "Current Frame" panel) are
untouched. Totals reset only when this object is recreated (backend restart) or
``reset()`` is called explicitly.

Appends to the payload:
    payload["statistics"] = {
        "frames": .., "vehicles": ..,               # frames processed, total unique
        "cars": .., "bikes": .., "cycles": .., "autos": ..,
        "buses": .., "trucks": .., "ambulances": ..,
    }
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import (
    COUNTING_LINE_AXIS,
    COUNTING_LINE_POSITION,
    NOMINAL_FPS,
    VEHICLE_TRACK_CONFIRMATION_SECONDS,
)

from .counter import CANONICAL_CLASSES, DEFAULT_ALIASES
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

# Stable numeric id per canonical class (required to feed the tracker).
_CLASS_INDEX: Dict[str, int] = {c: i for i, c in enumerate(CANONICAL_CLASSES)}
_ALIASES: Dict[str, str] = {k.lower(): v for k, v in DEFAULT_ALIASES.items()}


class TrafficStatistics:
    """Counts each unique vehicle once as it crosses a virtual counting line."""

    name = "statistics"

    def __init__(
        self,
        line_position: float = COUNTING_LINE_POSITION,
        line_axis: str = COUNTING_LINE_AXIS,
        confirmation_seconds: float = VEHICLE_TRACK_CONFIRMATION_SECONDS,
        fps: float = NOMINAL_FPS,
        default_size: Tuple[int, int] = (1920, 1080),
    ) -> None:
        """
        Args:
            line_position: counting-line location as a fraction (0..1) of the frame.
            line_axis: "y" -> horizontal line (counts vehicles moving vertically);
                       "x" -> vertical line (counts vehicles moving horizontally).
            confirmation_seconds: a track must be continuously tracked for at
                least this long before it becomes eligible to be counted.
            fps: reference FPS used to convert the confirmation window to frames.
            default_size: fallback (w, h) if a frame omits its dimensions.
        """
        self.line_position = float(line_position)
        self.line_axis = "x" if str(line_axis).lower() == "x" else "y"
        # Confirmation window in frames (e.g. 0.3s @ 30 FPS = 9 frames).
        self._confirm_frames = max(1, round(float(confirmation_seconds) * float(fps)))
        self._default_size = default_size

        self._frames = 0
        self._totals: Dict[str, int] = {STAT_KEYS[c]: 0 for c in CANONICAL_CLASSES}
        # counted_ids[stat_key] = set of track_ids already counted (never recount)
        self._counted_ids: Dict[str, set] = {STAT_KEYS[c]: set() for c in CANONICAL_CLASSES}
        # consecutive frames each track has been continuously observed
        self._track_age: Dict[int, int] = {}
        self._last_dims = default_size

        self._tracker = self._build_tracker()

    # -- setup / reset ------------------------------------------------------

    def _build_tracker(self):
        """Reuse the existing ByteTrack wrapper; fall back to None (no counting)
        only if the tracking dependency is unavailable, so the pipeline never
        crashes."""
        try:
            from config import TrackerConfig
            from tracker import VehicleTracker
            return VehicleTracker(TrackerConfig())
        except Exception as exc:  # pragma: no cover - dependency safety net
            logger.warning("Tracker unavailable (%s); cumulative counting disabled.", exc)
            return None

    def reset(self) -> None:
        """Explicit reset of all cumulative statistics (never called during
        normal playback)."""
        self._frames = 0
        for k in self._totals:
            self._totals[k] = 0
            self._counted_ids[k].clear()
        self._track_age.clear()
        logger.info("Session statistics reset.")

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _canonical(raw_class: Any) -> Optional[str]:
        return _ALIASES.get(str(raw_class).strip().lower())

    def _side(self, cx: float, cy: float) -> int:
        coord = cy if self.line_axis == "y" else cx
        return 1 if coord >= self.line_position else 0

    def _track(self, detections: List[Dict[str, Any]]):
        """Reconstruct Detection objects and run the existing tracker, returning
        (track_id, canonical_class, centroid_xy) tuples in pixel coordinates."""
        from detector import Detection

        dets: List[Detection] = []
        for det in detections:
            canonical = self._canonical(det.get("class", ""))
            if canonical is None:
                continue
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            dets.append(Detection(
                xyxy=np.asarray(bbox, dtype=np.float32),
                confidence=float(det.get("confidence", 0.0)),
                class_id=_CLASS_INDEX[canonical],
                class_name=canonical,
            ))
        tracked = self._tracker.update(dets)
        return [(t.track_id, t.class_name, float(t.confidence), t.centroid) for t in tracked]

    # -- consumer hook ------------------------------------------------------

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Update cumulative unique counts from confirmed, tracked line-crossings.

        Also publishes ``payload["tracked_objects"]`` (track_id + class +
        confidence + normalized centroid) so downstream stages (ambulance
        confirmation) reuse the very same ByteTrack IDs.
        """
        self._frames += 1
        tracked_objects: List[Dict[str, Any]] = []

        if self._tracker is not None:
            w = payload.get("frame_width") or self._last_dims[0]
            h = payload.get("frame_height") or self._last_dims[1]
            self._last_dims = (w, h)

            present_ids: set = set()
            for track_id, canonical, confidence, centroid in self._track(payload.get("detections", [])):
                present_ids.add(track_id)
                cx, cy = float(centroid[0]) / w, float(centroid[1]) / h
                # Stable-track confirmation: how many consecutive frames seen.
                self._track_age[track_id] = self._track_age.get(track_id, 0) + 1

                tracked_objects.append({
                    "track_id": int(track_id),
                    "class": canonical,
                    "confidence": round(confidence, 3),
                    "cx": cx,
                    "cy": cy,
                })

                stat_key = STAT_KEYS.get(canonical)
                if stat_key is None:
                    continue
                # Count once: ANY track confirmed as stable (tracked continuously
                # for >= confirm_frames). No counting-line crossing is required,
                # so vehicles in slow/congested traffic that never cross a line
                # are still counted. counted_ids ensures each track_id counts once.
                if (self._track_age[track_id] >= self._confirm_frames
                        and track_id not in self._counted_ids[stat_key]):
                    self._counted_ids[stat_key].add(track_id)
                    self._totals[stat_key] += 1

            # Tracks that vanished this frame must re-confirm if they return,
            # so a 1-2 frame flicker never yields a count.
            for tid in list(self._track_age):
                if tid not in present_ids:
                    self._track_age.pop(tid, None)

        payload["tracked_objects"] = tracked_objects
        payload["statistics"] = self.snapshot()
        return payload

    def snapshot(self) -> Dict[str, Any]:
        """Return the current cumulative statistics as a plain dict."""
        return {
            "frames": self._frames,
            "vehicles": sum(self._totals.values()),
            **self._totals,
        }
