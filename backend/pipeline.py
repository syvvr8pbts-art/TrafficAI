"""
backend/pipeline.py

The extensibility seam of the whole backend. It defines:

  * ``DetectionConsumer`` -- the contract every future analytics module implements
    (a single ``process(payload)`` method), and
  * ``AnalyticsPipeline`` -- a dispatcher that fans each frame's detection payload
    out to all registered consumers.

This is precisely what lets the planned modules -- counter, density, occupancy,
recommendation, ambulance, statistics, logger, snapshot -- be plugged in with a
single ``pipeline.register(...)`` call, WITHOUT modifying ``detector.py``. The
detector produces payloads; the pipeline routes them; consumers analyze them.
Separation of concerns is enforced by this one-directional data flow.
"""
from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable

from .utils import get_logger

logger = get_logger(__name__)


@runtime_checkable
class DetectionConsumer(Protocol):
    """Interface for any module that consumes per-frame detection payloads.

    Future modules only need a ``process`` method. A ``name`` attribute is
    optional and used purely for logging/introspection.
    """

    def process(self, payload: Dict[str, Any]) -> None:
        ...


class AnalyticsPipeline:
    """Registers detection consumers and dispatches payloads to them.

    A consumer failure is logged and isolated so one broken module cannot take
    down the frame loop or the other consumers.
    """

    def __init__(self) -> None:
        self._consumers: List[DetectionConsumer] = []

    def register(self, consumer: DetectionConsumer) -> "AnalyticsPipeline":
        """Add a consumer. Returns self so registrations can be chained."""
        self._consumers.append(consumer)
        logger.info("Registered consumer: %s", self._consumer_name(consumer))
        return self

    def consumers(self) -> List[str]:
        """Names of the currently registered consumers (for introspection)."""
        return [self._consumer_name(c) for c in self._consumers]

    def dispatch(self, payload: Dict[str, Any]) -> None:
        """Send one detection payload to every registered consumer."""
        for consumer in self._consumers:
            try:
                consumer.process(payload)
            except Exception:  # isolate a misbehaving consumer
                logger.exception("Consumer %s failed on frame %s",
                                 self._consumer_name(consumer), payload.get("frame_id"))

    @staticmethod
    def _consumer_name(consumer: DetectionConsumer) -> str:
        return getattr(consumer, "name", type(consumer).__name__)
