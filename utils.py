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
