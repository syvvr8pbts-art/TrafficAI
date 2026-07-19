"""
backend/ambulance.py

Ambulance priority / emergency override (signal-intelligence stage 4). Single
responsibility: raise a validated emergency and, while it holds, put the signal
controller into emergency mode -- GREEN for the ambulance's lane, RED for every
other lane -- then resume the normal cycle once it has passed.

Robust confirmation (prevents one-frame false positives). Emergency mode is
activated ONLY when a single tracked vehicle satisfies ALL of:
    1. detection confidence >= AMBULANCE_MIN_CONFIDENCE (config, default 0.80),
    2. it stays classified as "ambulance" for AMBULANCE_CONFIRMATION_SECONDS
       continuously (config, default 2.0s), AND
    3. it keeps the same ByteTrack ID for that whole window (not lost, not
       reclassified). If the track breaks / confidence drops / class changes,
       the confirmation timer resets and emergency mode is NOT activated.

It reuses the same ByteTrack IDs the statistics stage already produces, via
``payload["tracked_objects"]`` -- no second tracker, one consistent identity.
It coordinates with the controller via dependency injection.

Appends to the payload:
    payload["ambulance"] = {"active": bool, "lane": <lane or None>}
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from config import (
    AMBULANCE_CONFIRMATION_SECONDS,
    AMBULANCE_MIN_CONFIDENCE,
    NOMINAL_FPS,
)

from .utils import get_logger

logger = get_logger(__name__)


class AmbulancePriority:
    """Activates an emergency green corridor only for a confirmed ambulance."""

    name = "ambulance"

    def __init__(
        self,
        controller: Optional[Any] = None,
        lane_rois: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
        default_lane: str = "North",
        min_confidence: float = AMBULANCE_MIN_CONFIDENCE,
        confirmation_seconds: float = AMBULANCE_CONFIRMATION_SECONDS,
        fps: float = NOMINAL_FPS,
    ) -> None:
        self.controller = controller
        self.lane_rois = lane_rois or {}   # normalized [0,1] ROIs, keyed by lane
        self.default_lane = default_lane
        self.min_confidence = float(min_confidence)
        # Confirmation window in frames (e.g. 2.0s @ 30 FPS = 60 frames).
        self._confirm_frames = max(1, round(float(confirmation_seconds) * float(fps)))

        # track_id -> consecutive frames it has been a confident ambulance
        self._candidate_frames: Dict[int, int] = {}
        # the currently confirmed ambulance track holding emergency (or None)
        self._active_track: Optional[int] = None

    def _lane_for(self, cx: float, cy: float) -> str:
        """Infer the lane from normalized lane ROIs, else the default lane."""
        if not self.lane_rois:
            return self.default_lane
        for lane, (x1, y1, x2, y2) in self.lane_rois.items():
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return lane
        return self.default_lane

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Consumer hook: confirm an ambulance across time before any override."""
        tracked = payload.get("tracked_objects", [])
        # Ambulance tracks that currently pass the confidence gate, by track_id.
        ambulances: Dict[int, Dict[str, Any]] = {
            t["track_id"]: t
            for t in tracked
            if t.get("class") == "ambulance" and float(t.get("confidence", 0.0)) >= self.min_confidence
        }

        # Reset the timer for any candidate that is no longer a confident
        # ambulance (lost track / class change / confidence dropped).
        for tid in list(self._candidate_frames):
            if tid not in ambulances:
                del self._candidate_frames[tid]
        # Advance the timer for the candidates still present.
        for tid in ambulances:
            self._candidate_frames[tid] = self._candidate_frames.get(tid, 0) + 1

        active = False
        lane: Optional[str] = None

        if self._active_track is not None and self._active_track in ambulances:
            # Confirmed ambulance still present -> hold emergency on its lane.
            t = ambulances[self._active_track]
            lane = self._lane_for(t["cx"], t["cy"])
            active = True
            if self.controller is not None:
                self.controller.activate_emergency(lane)
        elif self._active_track is not None:
            # Confirmed ambulance left the scene -> resume the normal cycle once.
            if self.controller is not None:
                self.controller.deactivate_emergency()
            logger.info("Ambulance (track %s) cleared -> resuming normal cycle", self._active_track)
            self._active_track = None
        else:
            # Not in emergency: promote the first track confirmed for the full window.
            for tid, frames in self._candidate_frames.items():
                if frames >= self._confirm_frames:
                    t = ambulances[tid]
                    lane = self._lane_for(t["cx"], t["cy"])
                    active = True
                    self._active_track = tid
                    if self.controller is not None:
                        self.controller.activate_emergency(lane)
                    logger.info(
                        "Ambulance CONFIRMED (track %s, %d frames, conf %.2f) -> emergency on %s",
                        tid, frames, float(t.get("confidence", 0.0)), lane,
                    )
                    break

        payload["ambulance"] = {"active": active, "lane": lane}
        return payload
