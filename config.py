"""
config.py

Centralized configuration for the AI Traffic Flow Analysis system.
All tunable parameters live here so the rest of the codebase never
hard-codes values. Overridable per-run via CLI flags in main.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"
VIDEOS_DIR = BASE_DIR / "videos"
OUTPUT_DIR = BASE_DIR / "output"
CONFIGS_DIR = BASE_DIR / "configs"

# ---------------------------------------------------------------------------
# COCO class IDs (as produced by stock Ultralytics YOLO11 checkpoints) that
# we treat as "vehicles". auto_rickshaw and ambulance are NOT native COCO
# classes -- they are approximated by ambulance_detection.py using color/
# shape heuristics layered on top of these base detections.
# ---------------------------------------------------------------------------
VEHICLE_CLASS_MAP: Dict[int, str] = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

HEURISTIC_LABELS: List[str] = ["auto_rickshaw", "ambulance"]


@dataclass
class DetectorConfig:
    weights_path: str = str(WEIGHTS_DIR / "yolo11s.pt")
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    device: str = "cuda"  # "cuda", "cpu", or "cuda:0" etc.
    image_size: int = 640
    classes: List[int] = field(default_factory=lambda: list(VEHICLE_CLASS_MAP.keys()))


@dataclass
class TrackerConfig:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    frame_rate: int = 30


@dataclass
class AnalysisConfig:
    # Rough default lane area (sqm) used for occupancy calc when a lane's
    # own real_world_area_sqm isn't set in the lane config.
    density_area_sqm_per_lane: float = 50.0
    congestion_thresholds: Dict[str, float] = field(
        default_factory=lambda: {"green": 0.25, "yellow": 0.50, "orange": 0.75}
        # anything above "orange" => red
    )
    report_interval_minutes: int = 60


@dataclass
class PreprocessConfig:
    enable_night_enhancement: bool = False
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    frame_skip: int = 0  # process every (frame_skip + 1)th frame; 0 = every frame


@dataclass
class SignalConfig:
    """Bounds and fixed phases for a simulated traffic signal. The decision
    engine picks a green duration inside [min_green, max_green]; yellow and
    all-red (clearance) phases are fixed.
    """
    min_green_seconds: int = 10
    max_green_seconds: int = 60
    yellow_seconds: int = 3
    all_red_seconds: int = 2  # inter-phase clearance where every lane is red
    history_length: int = 20  # number of recent phase transitions to retain


@dataclass
class DecisionConfig:
    """Weights and knobs for the AI decision engine. The engine produces a
    per-lane `demand_score` in [0, 1] as a weighted blend of live occupancy
    ratio, normalized cumulative count, and congestion severity -- plus an
    optional vehicle-type term when per-lane type counts are available. The
    engine only ranks lanes; it does NOT compute signal timings (that is the
    signal controller's job).
    """
    # Base-metric weights. When the optional type term is available it is
    # folded in and the blend is renormalized over whichever terms are present,
    # so these do not need to sum to 1.0.
    occupancy_weight: float = 0.4   # weight of live occupancy ratio
    count_weight: float = 0.2       # weight of normalized cumulative count
    congestion_weight: float = 0.4  # weight of congestion-level severity
    type_weight: float = 0.3        # weight of vehicle-type intensity (optional)

    # Numeric severity for each congestion label emitted by TrafficAnalyzer.
    congestion_severity: Dict[str, float] = field(
        default_factory=lambda: {"green": 0.0, "yellow": 0.4, "orange": 0.7, "red": 1.0}
    )
    # Relative "pressure" each vehicle type contributes; heavier / higher
    # priority vehicles raise a lane's demand. Used only when per-lane type
    # counts are present in the analyzer summary.
    vehicle_type_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "bicycle": 0.3,
            "motorcycle": 0.5,
            "auto_rickshaw": 0.7,
            "car": 1.0,
            "bus": 2.0,
            "truck": 2.0,
            "ambulance": 3.0,
        }
    )
    # Lanes whose demand_score is below this are treated as effectively empty
    # for service-ordering purposes.
    empty_demand_threshold: float = 0.02


@dataclass
class AppConfig:
    video_source: str = str(VIDEOS_DIR / "sample.mp4")
    lanes_config_path: str = str(CONFIGS_DIR / "lanes_example.json")
    output_dir: str = str(OUTPUT_DIR)
    save_output_video: bool = True
    display_window: bool = True
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)


def load_config() -> AppConfig:
    """Returns the default application configuration. Values can be
    overridden by CLI flags in main.py without touching this file.
    """
    return AppConfig()
