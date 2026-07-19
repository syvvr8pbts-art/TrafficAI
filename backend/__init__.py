"""
backend package

Clean, modular serving layer for TrafficAI. It sits on top of the existing,
unchanged detection pipeline and exposes:

  * a structured per-frame detection payload (``FrameDetector`` / ``build_payload``),
  * an extensibility seam for future analytics (``AnalyticsPipeline`` /
    ``DetectionConsumer``), and
  * an HTTP API and entry point (``backend.api`` / ``backend.app``).

Design contract: the detector only produces structured detection data; every
future analytics module (counter, density, occupancy, recommendation, ambulance,
statistics, logger, snapshot) consumes that data via the pipeline and can be
added WITHOUT modifying ``detector.py``.

Note: ``backend.api`` is intentionally not imported here so that importing this
package never requires Flask. Run modules from the project root.
"""
from __future__ import annotations

from .ambulance import AmbulancePriority
from .clearance import IntersectionClearance
from .config import BackendConfig, ServerConfig, get_config
from .counter import VehicleCounter
from .density import TrafficDensity
from .detector import FrameDetector, build_payload, detection_to_dict
from .logger import DetectionLogger
from .occupancy import LaneOccupancy
from .pipeline import AnalyticsPipeline, DetectionConsumer
from .recommendation import RecommendationEngine
from .signal_controller import SignalController, SignalState
from .statistics import TrafficStatistics

__all__ = [
    "BackendConfig",
    "ServerConfig",
    "get_config",
    "FrameDetector",
    "build_payload",
    "detection_to_dict",
    "AnalyticsPipeline",
    "DetectionConsumer",
    # analytics consumers
    "VehicleCounter",
    "TrafficDensity",
    "LaneOccupancy",
    "TrafficStatistics",
    "DetectionLogger",
    # signal-intelligence consumers
    "RecommendationEngine",
    "SignalController",
    "SignalState",
    "IntersectionClearance",
    "AmbulancePriority",
]
