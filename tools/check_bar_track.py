"""Integration check for the optical-flow bar tracker (src/barbell.py).

Auto-seeds the plate (HSV detect on frame 0, near the pose anchor), runs track_plate with that
seed, prints drift stats, and renders the smoothed bar path on the clip so the path can be eyeballed.

Run:  .\.venv\Scripts\python.exe tools\check_bar_track.py input\squat-2.mov --lift squat
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, so `src` imports work

from src import barbell as B
from src import pose as P


def _speed_color(t: float):
    """Blue (slow) -> red (fast), BGR. t in [0, 1]."""
    return (int(255 * (1 - t)), 0, int(255 * t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--lift", default="squat", choices=["squat", "deadlift"])
    ap.add_argument("--backend", default="yolo")
    args = ap.parse_args()

    video = Path(args.video)
    pose = P.estimate_pose(str(video), backend=args.backend)
    n, fps = pose.num_frames, pose.fps
    anchor = B._anchor_series(pose, args.lift)

    # --- auto-seed: HSV plate detect on frame 0, nearest the bar anchor ---
    cap = cv2.VideoCapture(str(video))
    ok, frame0 = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("could not read frame 0")
    h = frame0.shape[0]
    min_r, max_r = int(h * 0.05), int(h * 0.25)
    hit = B._detect_plate(frame0, anchor[0], min_r, max_r, np.pi * min_r * min_r * 0.5)
    if hit is None:
        raise SystemExit("no plate found on frame 0 to seed from")
    seed = (float(hit[0][0]), float(hit[0][1]), float(hit[1]))
    print(f"seed (cx, cy, r) = ({seed[0]:.0f}, {seed[1]:.0f}, {seed[2]:.0f})  | frames={n} fps={fps:.0f}")

    # --- run the tracker under test ---
    centers, radii = B.track_plate(str(video), pose, args.lift, seed=seed)

    # --- drift stats: a clean track has small, steady frame-to-frame steps and no teleports ---
    step = np.hypot(np.diff(centers[:, 0]), np.diff(centers[:, 1]))
    print(f"frame-to-frame step px: median={np.median(step):.2f}  p95={np.percentile(step,95):.2f}  "
          f"max={np.max(step):.2f}")
    print(f"path y-span px = {np.nanmax(centers[:,1]) - np.nanmin(centers[:,1]):.0f}  "
          f"x-span px = {np.nanmax(centers[:,0]) - np.nanmin(centers[:,0]):.0f}")

    # --- render the path (speed-coloured) so we can SEE whether it stays glued to the plate ---
    norm = np.percentile(step, 95) or 1.0
    cap = cv2.VideoCapture(str(video))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_path = Path("output") / f"{video.stem}_klt_check.mp4"
    out_path.parent.mkdir(exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok or f >= n:
            break
        for i in range(1, f + 1):
            p0 = (int(centers[i - 1, 0]), int(centers[i - 1, 1]))
            p1 = (int(centers[i, 0]), int(centers[i, 1]))
            t = min(1.0, float(step[i - 1] / norm))
            cv2.line(frame, p0, p1, _speed_color(t), 2)
        cx, cy = int(centers[f, 0]), int(centers[f, 1])
        cv2.circle(frame, (cx, cy), int(radii[f]), (0, 255, 255), 2)  # current plate ring
        cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)
        vw.write(frame)
        f += 1
    cap.release()
    vw.release()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
