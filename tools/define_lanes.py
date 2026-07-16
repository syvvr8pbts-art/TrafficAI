"""
tools/define_lanes.py

Interactive helper: click points on a reference frame from your video to
define lane polygons. Controls:
    left-click  -> add a point to the lane currently being drawn
    n           -> finish current lane, start a new one
    s           -> save current lane (if 3+ points) and write the JSON file
    q           -> quit without saving

Produces a JSON file in the exact format LaneManager expects.

Usage:
    python tools/define_lanes.py --source ../videos/sample.mp4 --output ../configs/lanes_example.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_WINDOW = "Define Lanes  (click points | n=new lane | s=save | q=quit)"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Video file to grab a reference frame from")
    parser.add_argument("--output", required=True, help="Where to save the lane JSON config")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"Could not read a frame from {args.source}", file=sys.stderr)
        return 1

    lanes = []
    current_points = []
    lane_index = 1

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append([x, y])

    cv2.namedWindow(_WINDOW)
    cv2.setMouseCallback(_WINDOW, on_click)

    while True:
        display = frame.copy()
        for lane in lanes:
            cv2.polylines(display, [np.array(lane["points"], dtype=np.int32)], True, (255, 200, 0), 2)
        if len(current_points) > 1:
            cv2.polylines(display, [np.array(current_points, dtype=np.int32)], False, (0, 255, 0), 2)
        for pt in current_points:
            cv2.circle(display, tuple(pt), 4, (0, 0, 255), -1)

        cv2.imshow(_WINDOW, display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("n"):
            if len(current_points) >= 3:
                lanes.append({"name": f"lane_{lane_index}", "points": current_points})
                lane_index += 1
            current_points = []
        elif key == ord("s"):
            if len(current_points) >= 3:
                lanes.append({"name": f"lane_{lane_index}", "points": current_points})
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump({"lanes": lanes}, f, indent=2)
            print(f"Saved {len(lanes)} lane(s) to {args.output}")
            break
        elif key == ord("q"):
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
