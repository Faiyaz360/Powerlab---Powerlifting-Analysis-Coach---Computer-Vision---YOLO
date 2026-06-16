"""Render a clip with each pose backend and stack them side-by-side for a visual comparison.

    .\.venv\Scripts\python.exe tools/compare_backends.py input/squat-1.mov --lift squat

Writes output/compare/<name>_sidebyside.mp4 — the same clip with the MediaPipe, YOLO and RTMPose
skeletons drawn, panels labelled and placed left-to-right. Reuses the pose cache, so it's fast
once eval/run_eval has already run those backends on the clip.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import pipeline  # noqa: E402

BACKENDS = ["mediapipe", "yolo", "rtmpose"]
PANEL_H = 480  # each panel scaled to this height so the stacked file stays a sane size


def _render_each(video, lift):
    """Run the pipeline with each backend; return [(backend, annotated_path)] for the ones that work."""
    panels = []
    for b in BACKENDS:
        out_dir = ROOT / "output" / "compare" / b
        try:
            res = pipeline.analyze(video, lift=lift, out_dir=str(out_dir), backend=b)
            panels.append((b, res["paths"]["annotated_video"]))
            print(f"{b}: {res['paths']['annotated_video']}")
        except Exception as exc:  # a missing/failing backend shouldn't sink the others
            print(f"{b}: FAILED — {exc}")
    return panels


def _stack(panels, out_path):
    """Read the annotated clips in lockstep, label each, scale to a common height, hconcat, write."""
    caps = [(b, cv2.VideoCapture(p)) for b, p in panels]
    fps = caps[0][1].get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    while True:
        frames = []
        for b, cap in caps:
            ok, fr = cap.read()
            if not ok:
                frames = None
                break
            scale = PANEL_H / fr.shape[0]
            fr = cv2.resize(fr, (int(fr.shape[1] * scale), PANEL_H))
            cv2.putText(fr, b, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(fr, b, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            frames.append(fr)
        if not frames:
            break
        row = cv2.hconcat(frames)
        if writer is None:
            h, w = row.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(row)
    for _, cap in caps:
        cap.release()
    if writer is not None:
        writer.release()


def main():
    ap = argparse.ArgumentParser(description="Side-by-side pose-backend comparison video")
    ap.add_argument("video")
    ap.add_argument("--lift", required=True, choices=["squat", "deadlift"])
    args = ap.parse_args()

    panels = _render_each(args.video, args.lift)
    if len(panels) < 2:
        print("Need at least 2 backends to stack — nothing written.")
        return

    out = ROOT / "output" / "compare" / f"{Path(args.video).stem}_sidebyside.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    _stack(panels, out)
    print("\nside-by-side:", out)


if __name__ == "__main__":
    main()
