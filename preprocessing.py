"""
preprocessing.py

Frame-level preprocessing: optional low-light/night enhancement (CLAHE) and
frame-skip bookkeeping. Kept deliberately small -- heavier preprocessing
(deblurring, stabilization) can be added here later without touching any
other module.
"""
from __future__ import annotations

import cv2
import numpy as np

from config import PreprocessConfig


class FramePreprocessor:
    def __init__(self, cfg: PreprocessConfig) -> None:
        self.cfg = cfg
        self._clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip_limit,
            tileGridSize=(cfg.clahe_grid_size, cfg.clahe_grid_size),
        )
        self._frame_counter = 0

    def should_process(self) -> bool:
        """Frame-skip gate: returns False for frames that should be skipped.
        Advances the internal counter as a side effect, so call exactly
        once per incoming frame.
        """
        skip = self._frame_counter % (self.cfg.frame_skip + 1) != 0
        self._frame_counter += 1
        return not skip

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE-based contrast enhancement on the luminance channel
        only, to improve detection recall in low-light / night footage.
        """
        if not self.cfg.enable_night_enhancement:
            return frame
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_enhanced = self._clahe.apply(l_channel)
        lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
