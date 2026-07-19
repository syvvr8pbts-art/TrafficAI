"""
backend/streamer.py

Dashboard video processor. Runs the existing detection + analytics + signal
pipeline over a video source in a BACKGROUND thread, and exposes two things the
Flask dashboard consumes:

    * the latest annotated frame as JPEG bytes (for the MJPEG /video_feed), and
    * the latest full state via the TrafficSystem it feeds (for /state polling).

It is pure glue: it calls the already-complete FrameDetector and TrafficSystem
(no detection/analytics logic lives here) and only draws an overlay for display.
Nothing in the forbidden backend modules is touched.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .detector import FrameDetector
from .utils import get_logger

logger = get_logger(__name__)

# Overlay colors (BGR) per signal state and detection boxes.
_STATE_COLORS = {
    "GREEN": (0, 200, 0),
    "YELLOW": (0, 210, 210),
    "ALL_RED": (0, 0, 220),
    "CLEARANCE": (0, 120, 220),
}
_BOX_COLOR = (60, 200, 60)
_AMBULANCE_COLOR = (0, 0, 255)


class VideoProcessor:
    """Threaded processor turning a video source into annotated JPEG frames."""

    def __init__(
        self,
        source: str,
        system: Any,
        detector: Optional[FrameDetector] = None,
        loop: bool = True,
        jpeg_quality: int = 80,
        target_fps: float = 20.0,
    ) -> None:
        self._source = source
        self._system = system
        self._detector = detector or FrameDetector()
        self._loop = loop
        self._quality = int(jpeg_quality)
        self._min_dt = 1.0 / target_fps if target_fps > 0 else 0.0

        self._latest_jpeg: Optional[bytes] = self._placeholder_jpeg("Starting stream...")
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> "VideoProcessor":
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._run, name="VideoProcessor", daemon=True)
        self._thread.start()
        logger.info("VideoProcessor started on source: %s", self._source)
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # -- consumers ----------------------------------------------------------

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    # -- processing loop ----------------------------------------------------

    def _run(self) -> None:
        src = int(self._source) if str(self._source).isdigit() else self._source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            logger.error("VideoProcessor could not open source: %s", self._source)
            with self._lock:
                self._latest_jpeg = self._placeholder_jpeg("Cannot open video source")
            self._running = False
            return

        # Files report a positive frame count and are seekable; live cameras /
        # streams report 0 and cannot be rewound.
        seekable = cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
        consecutive_failures = 0

        while self._running:
            ok, frame = cap.read()
            if not ok:
                # Seekable clip reached its end -> loop it for a continuous demo.
                if self._loop and seekable:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    consecutive_failures = 0
                    continue
                # Non-seekable source failed/ended (e.g. camera disconnect): show
                # a placeholder and back off instead of busy-spinning the CPU.
                consecutive_failures += 1
                with self._lock:
                    self._latest_jpeg = self._placeholder_jpeg("Waiting for video source...")
                if not self._loop or consecutive_failures >= 40:  # ~20s -> give up
                    logger.warning("VideoProcessor: source unavailable, stopping.")
                    break
                time.sleep(0.5)
                continue
            consecutive_failures = 0

            t0 = time.perf_counter()
            # Run the existing, unchanged pipeline.
            payload = self._detector.process(frame)
            payload["frame_height"], payload["frame_width"] = frame.shape[0], frame.shape[1]
            self._system.process(payload)

            annotated = self._annotate(frame, payload)
            ok_enc, buf = cv2.imencode(
                ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality]
            )
            if ok_enc:
                with self._lock:
                    self._latest_jpeg = buf.tobytes()

            # Pace the loop so we don't peg the CPU faster than the stream needs.
            dt = time.perf_counter() - t0
            if self._min_dt > dt:
                time.sleep(self._min_dt - dt)

        cap.release()
        logger.info("VideoProcessor stopped.")

    # -- drawing ------------------------------------------------------------

    def _annotate(self, frame: np.ndarray, payload: Dict[str, Any]) -> np.ndarray:
        """Draw detection boxes and a compact signal banner onto the frame."""
        out = frame
        for det in payload.get("detections", []):
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cls = str(det.get("class", ""))
            color = _AMBULANCE_COLOR if cls.lower() == "ambulance" else _BOX_COLOR
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{cls} {det.get('confidence', 0):.2f}"
            cv2.putText(out, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        signal = payload.get("signal", {})
        state = signal.get("state", "")
        banner = _STATE_COLORS.get(state, (80, 80, 80))
        text = (
            f"{signal.get('lane', '-')}  {state}  {signal.get('countdown', 0)}s"
            f"   |  {payload.get('density', '-')}  "
            f"occ {payload.get('occupancy', 0)}%"
        )
        if payload.get("ambulance", {}).get("active"):
            text += "   | AMBULANCE PRIORITY"
            banner = _AMBULANCE_COLOR
        cv2.rectangle(out, (0, 0), (out.shape[1], 34), (20, 20, 20), -1)
        cv2.rectangle(out, (0, 0), (12, 34), banner, -1)
        cv2.putText(out, text, (22, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)
        return out

    @staticmethod
    def _placeholder_jpeg(message: str) -> bytes:
        """A dark placeholder frame shown before/while the stream is unavailable."""
        img = np.full((480, 854, 3), 18, dtype=np.uint8)
        cv2.putText(img, message, (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
        ok, buf = cv2.imencode(".jpg", img)
        return buf.tobytes() if ok else b""

    def mjpeg_frames(self):
        """Generator yielding multipart MJPEG chunks for a Flask streaming route."""
        boundary = b"--frame"
        while True:
            jpeg = self.get_jpeg()
            if jpeg is None:
                time.sleep(0.05)
                continue
            yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            if self._min_dt:
                time.sleep(self._min_dt)
