"""
backend/logger.py

Detection logger (analytics stage 5). Single responsibility: persist a one-row
CSV record per processed frame. It is a side-effect module -- it reads the fully
assembled payload and writes it, adding only the log file path back.

Each row: timestamp, frame_id, per-class counts, total, density, occupancy.
The log directory and file (with header) are created automatically on first use.

Appends to the payload:
    payload["log_path"] = <path to the CSV file>
"""
from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional

from .counter import CANONICAL_CLASSES
from .utils import get_logger

logger = get_logger(__name__)

DEFAULT_LOG_PATH = "logs/detections.csv"


class DetectionLogger:
    """Appends one CSV row per frame; creates the log file/dir if missing."""

    name = "logger"

    def __init__(self, path: str = DEFAULT_LOG_PATH, classes: Optional[List[str]] = None) -> None:
        self.path = path
        self.classes = list(classes or CANONICAL_CLASSES)
        self._ensure_file()

    def _columns(self) -> List[str]:
        return [
            "timestamp", "frame_id",
            *[f"count_{c}" for c in self.classes],
            "total", "density", "occupancy",
        ]

    def _ensure_file(self) -> None:
        """Create the log directory and write the header if the file is new/empty."""
        directory = os.path.dirname(self.path) or "."
        os.makedirs(directory, exist_ok=True)
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self._columns())
            logger.info("Created detection log: %s", self.path)

    def log(self, payload: Dict[str, Any]) -> str:
        """Append one row for this frame and return the log path."""
        counts = payload.get("counts", {})
        row = [
            payload.get("timestamp"),
            payload.get("frame_id"),
            *[counts.get(c, 0) for c in self.classes],
            payload.get("total", 0),
            payload.get("density"),
            payload.get("occupancy"),
        ]
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        return self.path

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Consumer hook: write the CSV row and record the log path."""
        payload["log_path"] = self.log(payload)
        return payload
