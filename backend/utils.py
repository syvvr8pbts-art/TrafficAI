"""
backend/utils.py

Small, stateless helper functions shared across the backend package. It holds
NO business logic (no counting, density, analytics) -- only formatting and
plumbing helpers, so any repeated glue code lives in exactly one place.

Logging is delegated to the project-root ``utils.setup_logger`` so the backend
and the existing pipeline share one consistent log format.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Iterable, List

# Reuse the root logger factory (absolute import -> top-level utils.py).
from utils import setup_logger as _root_setup_logger


def get_logger(name: str):
    """Return a configured logger using the project's shared logger factory."""
    return _root_setup_logger(name)


def epoch_now() -> float:
    """Current wall-clock time as epoch seconds (machine-friendly timestamp)."""
    return time.time()


def iso_now() -> str:
    """Current UTC time as an ISO-8601 string (human-friendly timestamp)."""
    return datetime.now(timezone.utc).isoformat()


def to_float_list(values: Iterable[Any]) -> List[float]:
    """Coerce an iterable (e.g. a numpy bbox array) into a plain list of floats.

    Used to make detection payloads JSON-serializable without altering values.
    """
    return [float(v) for v in values]


def is_detection_payload(payload: Any) -> bool:
    """Lightweight structural check that an object looks like a frame payload.

    Future consumers can use this to guard their input. It validates shape only,
    never semantics.
    """
    if not isinstance(payload, dict):
        return False
    if "frame_id" not in payload or "detections" not in payload:
        return False
    return isinstance(payload["detections"], list)
