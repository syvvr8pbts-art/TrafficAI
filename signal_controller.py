"""
signal_controller.py

Event-driven traffic signal controller, modeled as a finite state machine and
scoped to one Intersection. It is the ONLY module that converts priorities into
timings: it derives each lane's green duration from the demand carried on the
last TrafficDecision it received, and cycles GREEN -> YELLOW -> ALL_RED before
serving the next lane in the decision's service order.

Design notes:
  * Event-driven: `submit_decision()` feeds a new TrafficDecision (an event).
    The controller adopts the new service order at the next cycle boundary and
    uses the updated demands when it starts a lane's green -- it does NOT
    recompute timings every frame.
  * Time-driven advance: `tick(now)` only checks the clock and transitions when
    the current phase has elapsed, so callers can tick at any (even irregular)
    frame rate and get correct countdowns.
  * Modes: NORMAL (adaptive cycle), EMERGENCY (hold green on one lane for a
    green corridor), MANUAL (operator hold). EMERGENCY/MANUAL freeze auto-advance.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

from config import SignalConfig
from decision_engine import TrafficDecision
from intersection import Intersection
from utils import setup_logger

logger = setup_logger(__name__)


class SignalPhase(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    ALL_RED = "all_red"


class ControlMode(str, Enum):
    NORMAL = "normal"
    EMERGENCY = "emergency"
    MANUAL = "manual"


@dataclass(frozen=True)
class SignalState:
    """Immutable snapshot of the controller at one instant."""
    intersection_id: str
    mode: str
    current_lane: Optional[str]
    current_phase: str
    remaining_seconds: float
    phase_duration_seconds: float
    next_lane: Optional[str]
    service_order: Tuple[str, ...]
    history: Tuple[Dict[str, object], ...] = ()

    def to_dict(self) -> Dict[str, object]:
        return {
            "intersection_id": self.intersection_id,
            "mode": self.mode,
            "current_lane": self.current_lane,
            "current_phase": self.current_phase,
            "remaining_seconds": round(self.remaining_seconds, 1),
            "phase_duration_seconds": round(self.phase_duration_seconds, 1),
            "next_lane": self.next_lane,
            "service_order": list(self.service_order),
            "history": list(self.history),
        }


class SignalController:
    """FSM signal controller for a single intersection."""

    def __init__(self, intersection: Intersection, cfg: SignalConfig, now: float = 0.0) -> None:
        self.intersection = intersection
        self.cfg = cfg

        self._mode = ControlMode.NORMAL
        # Active serving order and the pending order adopted at the next wrap.
        self._cycle: List[str] = list(intersection.lane_names) or [intersection.intersection_id]
        self._pending_cycle: List[str] = list(self._cycle)
        self._pos = 0
        # Latest known demand per lane (drives green duration); starts empty.
        self._demands: Dict[str, float] = {}
        self._held_lane: Optional[str] = None  # lane forced green in EMERGENCY/MANUAL

        self._history: Deque[Dict[str, object]] = deque(maxlen=cfg.history_length)

        self._phase = SignalPhase.GREEN
        self._phase_start = now
        self._phase_duration = self._duration_for(SignalPhase.GREEN)
        self._record(now)

    # -- lane helpers -------------------------------------------------------

    @property
    def _current_lane(self) -> Optional[str]:
        if not self._cycle:
            return None
        return self._cycle[self._pos % len(self._cycle)]

    def _next_lane(self) -> Optional[str]:
        if not self._cycle:
            return None
        return self._cycle[(self._pos + 1) % len(self._cycle)]

    # -- timing -------------------------------------------------------------

    def _green_seconds(self, lane: Optional[str]) -> float:
        """Green duration interpolated from the lane's last-known demand score."""
        score = self._demands.get(lane, 0.0) if lane is not None else 0.0
        span = self.cfg.max_green_seconds - self.cfg.min_green_seconds
        return float(self.cfg.min_green_seconds + max(0.0, min(score, 1.0)) * span)

    def _duration_for(self, phase: SignalPhase) -> float:
        if phase is SignalPhase.GREEN:
            return self._green_seconds(self._current_lane)
        if phase is SignalPhase.YELLOW:
            return float(self.cfg.yellow_seconds)
        return float(self.cfg.all_red_seconds)

    # -- transitions --------------------------------------------------------

    def _record(self, at: float) -> None:
        self._history.append({
            "time": round(at, 2),
            "lane": self._current_lane,
            "phase": self._phase.value,
            "duration": round(self._phase_duration, 1),
            "mode": self._mode.value,
        })

    def _enter_phase(self, phase: SignalPhase, at: float) -> None:
        self._phase = phase
        self._phase_start = at
        self._phase_duration = self._duration_for(phase)
        self._record(at)

    def _advance(self, at: float) -> None:
        """One boundary transition in NORMAL mode."""
        if self._phase is SignalPhase.GREEN:
            self._enter_phase(SignalPhase.YELLOW, at)
        elif self._phase is SignalPhase.YELLOW:
            self._enter_phase(SignalPhase.ALL_RED, at)
        else:  # ALL_RED -> serve next lane
            self._pos += 1
            if self._pos >= len(self._cycle):
                self._pos = 0
                self._cycle = list(self._pending_cycle)  # adopt latest order at wrap
            self._enter_phase(SignalPhase.GREEN, at)

    # -- public API ---------------------------------------------------------

    def submit_decision(self, decision: TrafficDecision, now: float) -> None:
        """Consume a TrafficDecision event: refresh demands and queue the new
        service order for adoption at the next cycle wrap. Does not interrupt
        the current phase.
        """
        self._demands = {
            lane: d.demand_score for lane, d in decision.demands.items()
        }
        order = decision.service_order or list(self.intersection.lane_names)
        self._pending_cycle = order or list(self._cycle)
        logger.debug(
            "Controller[%s] received decision: order=%s",
            self.intersection.intersection_id, order,
        )

    def force_emergency(self, lane: str, now: float) -> None:
        """Enter EMERGENCY mode and hold green on `lane` (green corridor)."""
        self._mode = ControlMode.EMERGENCY
        self._held_lane = lane
        self._point_cycle_at(lane)
        self._enter_phase(SignalPhase.GREEN, now)
        logger.info("Controller[%s] EMERGENCY green held on %s",
                    self.intersection.intersection_id, lane)

    def clear_emergency(self, now: float) -> None:
        """Leave EMERGENCY mode, ending the held green safely via YELLOW."""
        if self._mode is not ControlMode.EMERGENCY:
            return
        self._mode = ControlMode.NORMAL
        self._held_lane = None
        self._enter_phase(SignalPhase.YELLOW, now)
        logger.info("Controller[%s] EMERGENCY cleared -> resuming normal cycle",
                    self.intersection.intersection_id)

    def set_manual(self, lane: str, now: float) -> None:
        """Operator override: hold green on `lane` in MANUAL mode."""
        self._mode = ControlMode.MANUAL
        self._held_lane = lane
        self._point_cycle_at(lane)
        self._enter_phase(SignalPhase.GREEN, now)

    def clear_manual(self, now: float) -> None:
        if self._mode is not ControlMode.MANUAL:
            return
        self._mode = ControlMode.NORMAL
        self._held_lane = None
        self._enter_phase(SignalPhase.YELLOW, now)

    def _point_cycle_at(self, lane: str) -> None:
        if lane in self._cycle:
            self._pos = self._cycle.index(lane)
        else:
            self._cycle = [lane] + self._cycle
            self._pos = 0

    def tick(self, now: float) -> SignalState:
        """Advance the FSM to `now` and return the current immutable state.

        In EMERGENCY/MANUAL the green is held (no auto-advance). In NORMAL the
        machine transitions through as many boundaries as `now` has passed,
        which keeps timing correct even under frame skipping.
        """
        if self._mode is ControlMode.NORMAL:
            # Guard against pathological zero/negative durations.
            guard = 0
            while now >= self._phase_start + self._phase_duration and guard < 1000:
                boundary = self._phase_start + self._phase_duration
                self._advance(boundary)
                guard += 1
        return self.state(now)

    def state(self, now: float) -> SignalState:
        elapsed = now - self._phase_start
        remaining = max(0.0, self._phase_duration - elapsed)
        held = self._mode in (ControlMode.EMERGENCY, ControlMode.MANUAL)
        return SignalState(
            intersection_id=self.intersection.intersection_id,
            mode=self._mode.value,
            current_lane=self._current_lane,
            current_phase=self._phase.value,
            remaining_seconds=remaining,
            phase_duration_seconds=self._phase_duration,
            next_lane=self._held_lane if held else self._next_lane(),
            service_order=tuple(self._cycle),
            history=tuple(self._history),
        )
