"""
backend/signal_controller.py

Traffic signal controller (signal-intelligence stage 2) -- the heart of the
system. Single responsibility: run a safe traffic-light state machine for one
intersection that cycles green through the lanes one at a time.

State machine (per lane, in order):
    GREEN --> YELLOW --> ALL_RED --> CLEARANCE --> (next lane) GREEN

Safety guarantees:
    * Only ONE lane is ever GREEN.
    * The machine never goes GREEN -> GREEN directly; every green is followed by
      YELLOW, ALL_RED, and CLEARANCE before the next lane turns green.

Durations:
    * GREEN     = the recommended green time from the payload (adaptive).
    * YELLOW    = configurable (default 3s).
    * ALL_RED   = configurable (default 2s).
    * CLEARANCE = gated by the clearance module when intersection-occupancy info
      is available (see ``report_clearance``); otherwise a configurable timer.

The controller is time-driven from the payload timestamp, so countdowns are in
real seconds. It exposes a small public API used by the clearance and ambulance
modules (dependency injection), but it does NOT import them -- keeping it
independent and testable.

Appends to the payload:
    payload["signal"] = {"lane": .., "state": .., "countdown": .., "next_lane": ..}
"""
from __future__ import annotations

import math
import time
from enum import Enum
from typing import Any, Dict, List, Optional

from .utils import get_logger

logger = get_logger(__name__)

DEFAULT_LANES: List[str] = ["North", "East", "South", "West"]


class SignalState(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    ALL_RED = "ALL_RED"
    CLEARANCE = "CLEARANCE"


class SignalController:
    """A safe, adaptive, single-intersection traffic-light state machine."""

    name = "signal_controller"

    def __init__(
        self,
        lanes: Optional[List[str]] = None,
        yellow_seconds: float = 3.0,
        all_red_seconds: float = 2.0,
        clearance_seconds: float = 2.0,
        default_green_seconds: float = 15.0,
        emergency_green_seconds: int = 30,
    ) -> None:
        self.lanes = list(lanes or DEFAULT_LANES)
        self.yellow_seconds = yellow_seconds
        self.all_red_seconds = all_red_seconds
        self.clearance_seconds = clearance_seconds  # used only if no ROI info
        self.default_green_seconds = default_green_seconds
        self.emergency_green_seconds = emergency_green_seconds

        # Runtime state
        self._index = 0
        self.state = SignalState.GREEN
        self._phase_start: Optional[float] = None
        self._green_dur = default_green_seconds
        self._last_now = 0.0

        # Clearance gating (set by the clearance module via report_clearance)
        self._clearance_info = False
        self._clearance_blocked = False

        # Emergency override (set by the ambulance module)
        self._emergency = False
        self._emergency_lane: Optional[str] = None
        self._saved_index = 0

        # Exposed snapshot fields
        self.current_lane = self.lanes[0]
        self.current_signal = self.state.value
        self.countdown = 0
        self.next_lane = self._lane_at(1)

    # -- lane helpers -------------------------------------------------------

    def _lane_at(self, offset: int) -> str:
        return self.lanes[(self._index + offset) % len(self.lanes)]

    # -- public API used by clearance / ambulance (dependency injection) ----

    def report_clearance(self, blocked: bool) -> None:
        """Called by the clearance module: whether the intersection ROI is
        occupied. Enables ROI-based gating of the CLEARANCE phase.
        """
        self._clearance_info = True
        self._clearance_blocked = blocked

    def activate_emergency(self, lane: str) -> None:
        """Called by the ambulance module: hold GREEN on ``lane`` and pause the
        normal cycle. Idempotent while active (lane may be refreshed).
        """
        if not self._emergency:
            self._saved_index = self._index  # remember where to resume
            logger.info("Signal EMERGENCY engaged for lane %s", lane)
        self._emergency = True
        self._emergency_lane = lane

    def deactivate_emergency(self) -> None:
        """Called by the ambulance module: resume the normal cycle safely.

        We restore the pre-emergency lane and drop into ALL_RED so the machine
        never jumps straight back into a conflicting GREEN.
        """
        if not self._emergency:
            return
        self._emergency = False
        self._emergency_lane = None
        self._index = self._saved_index
        self.state = SignalState.ALL_RED
        self._phase_start = self._last_now
        logger.info("Signal EMERGENCY cleared -> resuming normal cycle")

    # -- transitions --------------------------------------------------------

    def _enter_green(self, now: float, payload: Dict[str, Any]) -> None:
        self.state = SignalState.GREEN
        self._phase_start = now
        # Adaptive green: use this frame's recommendation, captured at green start.
        self._green_dur = float(payload.get("recommended_green", self.default_green_seconds))

    def _to_state(self, state: SignalState, now: float) -> None:
        self.state = state
        self._phase_start = now

    def _phase_duration(self) -> float:
        return {
            SignalState.GREEN: self._green_dur,
            SignalState.YELLOW: self.yellow_seconds,
            SignalState.ALL_RED: self.all_red_seconds,
            SignalState.CLEARANCE: self.clearance_seconds,
        }[self.state]

    def _clearance_ready(self, elapsed: float) -> bool:
        """CLEARANCE -> next green gate.

        Prefer real occupancy info when available (do not use a fixed timer);
        otherwise fall back to the clearance timer.
        """
        if self._clearance_info:
            return not self._clearance_blocked
        return elapsed >= self.clearance_seconds

    def _advance(self, now: float, payload: Dict[str, Any]) -> None:
        start = self._phase_start if self._phase_start is not None else now
        elapsed = now - start
        if self.state is SignalState.GREEN:
            if elapsed >= self._green_dur:
                self._to_state(SignalState.YELLOW, now)
        elif self.state is SignalState.YELLOW:
            if elapsed >= self.yellow_seconds:
                self._to_state(SignalState.ALL_RED, now)
        elif self.state is SignalState.ALL_RED:
            if elapsed >= self.all_red_seconds:
                self._to_state(SignalState.CLEARANCE, now)
        elif self.state is SignalState.CLEARANCE:
            if self._clearance_ready(elapsed):
                self._index = (self._index + 1) % len(self.lanes)
                self._enter_green(now, payload)

    # -- consumer hook ------------------------------------------------------

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        ts = payload.get("timestamp")
        now = float(ts) if ts is not None else time.time()
        self._last_now = now
        if self._phase_start is None:  # first frame: start on the first lane's green
            self._enter_green(now, payload)

        if self._emergency:
            self._write_emergency_signal(payload)
        else:
            self._advance(now, payload)
            self._write_signal(payload, now)
        return payload

    def _write_signal(self, payload: Dict[str, Any], now: float) -> None:
        start = self._phase_start if self._phase_start is not None else now
        elapsed = now - start
        remaining = max(0, math.ceil(self._phase_duration() - elapsed))
        self.current_lane = self.lanes[self._index]
        self.current_signal = self.state.value
        self.countdown = remaining
        self.next_lane = self._lane_at(1)
        payload["signal"] = {
            "lane": self.current_lane,
            "state": self.current_signal,
            "countdown": self.countdown,
            "next_lane": self.next_lane,
        }

    def _write_emergency_signal(self, payload: Dict[str, Any]) -> None:
        # Held green on the emergency lane; countdown shows the hold budget.
        self.current_lane = self._emergency_lane or self.lanes[self._index]
        self.current_signal = SignalState.GREEN.value
        self.countdown = self.emergency_green_seconds
        self.next_lane = self.current_lane
        payload["signal"] = {
            "lane": self.current_lane,
            "state": self.current_signal,
            "countdown": self.countdown,
            "next_lane": self.next_lane,
            "emergency": True,
        }

    def snapshot(self) -> Dict[str, Any]:
        """Current signal state, for the API/terminal (no side effects)."""
        return {
            "lane": self.current_lane,
            "state": self.current_signal,
            "countdown": self.countdown,
            "next_lane": self.next_lane,
        }
