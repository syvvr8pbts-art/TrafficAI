"""
main.py

Entry point for the AI Traffic Flow Analysis pipeline. Reads a video
source, runs detection -> tracking -> lane assignment -> analytics ->
visualization per frame, optionally writes an annotated output video, and
saves a final traffic report (JSON + CSV) on exit.

Usage:
    python main.py --source videos/sample.mp4 --lanes configs/lanes_example.json
    python main.py --source 0 --no-save-video           # webcam, no video output
    python main.py --source videos/night_clip.mp4 --night-enhance --device cpu
"""
from __future__ import annotations

import argparse
import json
import signal
import sys

import cv2

from ambulance_detection import refine_label
from config import load_config
from detector import VehicleDetector
from lane_detection import LaneManager
from preprocessing import FramePreprocessor
from tracker import VehicleTracker
from traffic_analysis import TrafficAnalyzer
from utils import FPSTracker, ensure_dir, save_json, setup_logger
from visualization import draw_hud, draw_lanes, draw_tracks

logger = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="AI Traffic Flow Analysis")
    parser.add_argument("--source", default=cfg.video_source, help="Video file path, RTSP URL, or webcam index")
    parser.add_argument("--weights", default=cfg.detector.weights_path, help="Path to YOLO weights (.pt)")
    parser.add_argument("--lanes", default=cfg.lanes_config_path, help="Path to lane polygon JSON config")
    parser.add_argument("--output-dir", default=cfg.output_dir, help="Directory for saved video/report output")
    parser.add_argument("--device", default=cfg.detector.device, choices=["auto", "cuda", "mps", "cpu"], help="Inference device (auto picks CUDA > MPS > CPU)")
    parser.add_argument("--conf", type=float, default=cfg.detector.confidence_threshold, help="Detection confidence threshold")
    parser.add_argument("--night-enhance", action="store_true", help="Enable CLAHE low-light enhancement")
    parser.add_argument("--frame-skip", type=int, default=cfg.preprocess.frame_skip, help="Process every Nth+1 frame")
    parser.add_argument("--no-save-video", dest="save_video", action="store_false", help="Disable annotated video output")
    parser.add_argument("--no-display", dest="display", action="store_false", help="Disable live preview window")
    parser.set_defaults(save_video=cfg.save_output_video, display=cfg.display_window)
    return parser.parse_args()


def build_video_capture(source: str) -> cv2.VideoCapture:
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return cap


def main() -> int:
    args = parse_args()
    cfg = load_config()

    cfg.video_source = args.source
    cfg.detector.weights_path = args.weights
    cfg.detector.device = args.device
    cfg.detector.confidence_threshold = args.conf
    cfg.lanes_config_path = args.lanes
    cfg.output_dir = args.output_dir
    cfg.preprocess.enable_night_enhancement = args.night_enhance
    cfg.preprocess.frame_skip = args.frame_skip
    cfg.save_output_video = args.save_video
    cfg.display_window = args.display

    output_dir = ensure_dir(cfg.output_dir)

    logger.info("Initializing pipeline...")
    detector = VehicleDetector(cfg.detector)
    tracker = VehicleTracker(cfg.tracker)
    lane_manager = LaneManager(cfg.lanes_config_path)
    analyzer = TrafficAnalyzer(cfg.analysis, lane_manager)
    preprocessor = FramePreprocessor(cfg.preprocess)
    fps_tracker = FPSTracker()

    cap = build_video_capture(cfg.video_source)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if cfg.save_output_video:
        out_path = output_dir / "annotated_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_in, (width, height + 90))
        logger.info("Writing annotated video to %s", out_path)

    stop_flag = {"stop": False}

    def _handle_sigint(signum, frame):
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _handle_sigint)

    frame_idx = 0
    try:
        while cap.isOpened() and not stop_flag["stop"]:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            if not preprocessor.should_process():
                continue

            frame = preprocessor.enhance(frame)
            detections = detector.detect(frame)
            tracked_objects = tracker.update(detections)

            for obj in tracked_objects:
                obj.class_name = refine_label(frame, obj.xyxy, obj.class_name)

            lane_of = {obj.track_id: lane_manager.assign_lane(obj.centroid) for obj in tracked_objects}
            analyzer.update(tracked_objects, lane_of)

            fps = fps_tracker.tick()
            annotated = draw_lanes(frame, lane_manager)
            annotated = draw_tracks(annotated, tracked_objects)
            annotated = draw_hud(annotated, analyzer.summary(), fps)

            if cfg.display_window:
                cv2.imshow("AI Traffic Flow Analysis", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if writer is not None:
                writer.write(annotated)

            if frame_idx % 100 == 0:
                logger.info(
                    "Processed %d frames | FPS=%.1f | vehicles=%d",
                    frame_idx, fps, analyzer.summary()["total_unique_vehicles"],
                )

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

        summary = analyzer.summary()
        save_json(summary, output_dir / "traffic_summary.json")
        analyzer.export_csv(str(output_dir))
        logger.info("Final summary: %s", json.dumps(summary, indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())
