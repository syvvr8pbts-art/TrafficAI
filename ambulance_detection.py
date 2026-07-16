"""
ambulance_detection.py

Stock YOLO/COCO checkpoints have no "ambulance" or "auto_rickshaw" classes,
so this module re-classifies a subset of base detections ("car" and
"motorcycle") using color and shape heuristics as an approximate proxy.

This is intentionally simple and WILL misclassify vehicles that happen to
share a paint scheme or proportions with the target -- treat its output as
indicative, not ground truth. Swapping this out for a small fine-tuned
classifier head is the natural upgrade path if accuracy matters.
"""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

# HSV ranges used to look for the red/white (or red/blue) livery common on
# ambulances. Loose by design -- tune per region/fleet as needed.
_RED_HSV_RANGES: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = [
    ((0, 70, 50), (10, 255, 255)),
    ((170, 70, 50), (180, 255, 255)),
]
_WHITE_HSV_RANGE = ((0, 0, 200), (180, 40, 255))
_BLUE_HSV_RANGE = ((100, 70, 50), (130, 255, 255))
_YELLOW_HSV_RANGE = ((20, 80, 80), (35, 255, 255))
_GREEN_HSV_RANGE = ((35, 40, 40), (85, 255, 255))


def _color_fraction(hsv_crop: np.ndarray, ranges: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]) -> float:
    """Fraction of pixels in hsv_crop matching ANY of the given HSV ranges."""
    mask_total = np.zeros(hsv_crop.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask = cv2.inRange(hsv_crop, np.array(lower), np.array(upper))
        mask_total = cv2.bitwise_or(mask_total, mask)
    return float(np.count_nonzero(mask_total)) / max(mask_total.size, 1)


def looks_like_ambulance(bgr_crop: np.ndarray) -> bool:
    """Heuristic: significant red+white (or red+blue) coverage on the vehicle."""
    if bgr_crop.size == 0:
        return False
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    red_frac = _color_fraction(hsv, _RED_HSV_RANGES)
    white_frac = _color_fraction(hsv, [_WHITE_HSV_RANGE])
    blue_frac = _color_fraction(hsv, [_BLUE_HSV_RANGE])
    return (red_frac > 0.08 and white_frac > 0.15) or (red_frac > 0.08 and blue_frac > 0.15)


def looks_like_auto_rickshaw(bgr_crop: np.ndarray, xyxy: Tuple[float, float, float, float]) -> bool:
    """Heuristic: three-wheelers tend to be narrower relative to height than
    sedans, and are frequently yellow/green in many regions. Expect false
    positives on small hatchbacks and motorcycles carrying boxes.
    """
    if bgr_crop.size == 0:
        return False
    x1, y1, x2, y2 = xyxy
    width, height = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    aspect_ratio = width / height

    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    yellow_frac = _color_fraction(hsv, [_YELLOW_HSV_RANGE])
    green_frac = _color_fraction(hsv, [_GREEN_HSV_RANGE])

    narrow_and_tall = 0.6 <= aspect_ratio <= 1.15
    strong_color = yellow_frac > 0.12 or green_frac > 0.12
    return narrow_and_tall and strong_color


def refine_label(bgr_frame: np.ndarray, xyxy: np.ndarray, base_class_name: str) -> str:
    """Given a base COCO label ("car" or "motorcycle"), check heuristics and
    optionally upgrade the label to "ambulance" or "auto_rickshaw". All other
    classes (bus, truck, bicycle) pass through untouched.
    """
    if base_class_name not in ("car", "motorcycle"):
        return base_class_name

    h, w = bgr_frame.shape[:2]
    x1, y1, x2, y2 = [int(max(v, 0)) for v in xyxy]
    x2, y2 = min(x2, w), min(y2, h)
    crop = bgr_frame[y1:y2, x1:x2]

    if looks_like_ambulance(crop):
        return "ambulance"
    if base_class_name == "car" and looks_like_auto_rickshaw(crop, tuple(xyxy)):
        return "auto_rickshaw"
    return base_class_name
