"""
detector.py

Wraps an Ultralytics YOLO11 model to produce per-frame vehicle detections.
This module's only job is detection -- identity/tracking is handled by
tracker.py -- so it stays easy to swap out (different checkpoint, ONNX
export, etc.) without touching the rest of the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from config import DetectorConfig, VEHICLE_CLASS_MAP
from utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class Detection:
    """A single detected object in one frame."""
    xyxy: np.ndarray  # [x1, y1, x2, y2] float32, pixel coordinates
    confidence: float
    class_id: int
    class_name: str


class VehicleDetector:
    """Thin wrapper around ultralytics.YOLO restricted to vehicle classes."""

    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg
        self._model = self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required. Install with: pip install ultralytics"
            ) from exc

        logger.info("Loading YOLO weights from %s (device=%s)", self.cfg.weights_path, self.cfg.device)
        return YOLO(self.cfg.weights_path)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a single BGR frame and return vehicle detections only."""
        results = self._model.predict(
            source=frame,
            conf=self.cfg.confidence_threshold,
            iou=self.cfg.iou_threshold,
            imgsz=self.cfg.image_size,
            classes=self.cfg.classes,
            device=self.cfg.device,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(xyxy, confs, cls_ids):
            class_name = VEHICLE_CLASS_MAP.get(int(cls_id), f"class_{cls_id}")
            detections.append(
                Detection(
                    xyxy=box.astype(np.float32),
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=class_name,
                )
            )
        return detections
