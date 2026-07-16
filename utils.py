"""
utils.py

Small, dependency-light helpers shared across modules: logging setup,
directory helpers, JSON load/save, and a rolling FPS counter.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union


def setup_logger(name: str = "traffic_ai", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


_logger = setup_logger(__name__)


def resolve_device(requested: str = "auto") -> str:
    """Resolve a requested inference device to a concrete PyTorch device string.

    This is the single, centralized place where hardware acceleration is
    detected -- no other module should perform platform or capability checks.
    Detection relies solely on PyTorch capability probes (never on the host
    OS), so the same codebase runs unmodified across CUDA, Apple MPS, and CPU.

    Behavior:
        "auto" -> best available device, priority CUDA > MPS > CPU
        "cuda" -> CUDA if available, else best available (with a warning)
        "mps"  -> Apple MPS if available, else best available (with a warning)
        "cpu"  -> CPU (always available)

    Args:
        requested: One of "auto", "cuda", "mps", "cpu" (case-insensitive).
            Device suffixes like "cuda:0" are accepted and honored when CUDA
            is available.

    Returns:
        A concrete device string ("cuda", "mps", or "cpu") safe to hand to
        Ultralytics / PyTorch. Falls back to "cpu" if PyTorch is unavailable.
    """
    requested = (requested or "auto").strip().lower()

    try:
        import torch
    except ImportError:
        _logger.warning("PyTorch not importable; defaulting device to 'cpu'.")
        return "cpu"

    cuda_ok = bool(torch.cuda.is_available())
    mps_ok = bool(
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    )

    def _best_available() -> str:
        if cuda_ok:
            return "cuda"
        if mps_ok:
            return "mps"
        return "cpu"

    # Honor an explicit, indexed CUDA request (e.g. "cuda:0") when possible.
    if requested.startswith("cuda"):
        if cuda_ok:
            return requested
        _logger.warning("CUDA requested but unavailable; falling back.")
        return _best_available()

    if requested == "mps":
        if mps_ok:
            return "mps"
        _logger.warning("MPS requested but unavailable; falling back.")
        return _best_available()

    if requested == "cpu":
        return "cpu"

    if requested != "auto":
        _logger.warning("Unknown device '%s'; using auto-detection.", requested)
    return _best_available()


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


class FPSTracker:
    """Rolling FPS counter based on an exponential moving average."""

    def __init__(self, smoothing: float = 0.9) -> None:
        self._smoothing = smoothing
        self._fps = 0.0
        self._last_ts: Optional[float] = None

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last_ts is not None:
            instant_fps = 1.0 / max(now - self._last_ts, 1e-6)
            self._fps = (
                instant_fps
                if self._fps == 0.0
                else self._smoothing * self._fps + (1 - self._smoothing) * instant_fps
            )
        self._last_ts = now
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps
