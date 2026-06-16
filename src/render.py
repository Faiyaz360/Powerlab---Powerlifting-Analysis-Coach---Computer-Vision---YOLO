"""Draw the analysis onto the video: side skeleton, primary joint angle, rep counter, badge.

Lift-agnostic: it reads ``analysis['primary_key']`` (knee for squat, hip for deadlift) and the
per-rep ``badge`` / ``badge_frame``, so squat and deadlift share the same renderer.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import pose as P

GREEN = (0, 255, 0)
RED = (0, 0, 255)
ORANGE = (0, 200, 255)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
MAGENTA = (255, 0, 255)
SKELETON = (230, 230, 230)  # faint full-body skeleton, under the bold analysis chain

# which landmark labels the primary angle, per side
_PRIMARY_JOINT = {
    "knee": (P.L_KNEE, P.R_KNEE),
    "hip": (P.L_HIP, P.R_HIP),
}


def _xy(landmarks, f, idx):
    p = landmarks[f, idx, :2]
    if np.any(np.isnan(p)):
        return None
    return int(p[0]), int(p[1])


def render_video(in_path, out_path, pose: P.PoseResult, analysis: dict):
    """Write an annotated mp4 to ``out_path``."""
    lm = pose.landmarks
    side = analysis["series"]["side"]
    primary_key = analysis["primary_key"]
    primary = analysis["series"][primary_key]
    reps = analysis["reps"]
    rep_metrics = analysis["rep_metrics"]
    bar_xy = analysis.get("bar_xy")

    if side == "left":
        chain = [P.L_SHOULDER, P.L_HIP, P.L_KNEE, P.L_ANKLE, P.L_FOOT]
        joint_idx = _PRIMARY_JOINT[primary_key][0]
    else:
        chain = [P.R_SHOULDER, P.R_HIP, P.R_KNEE, P.R_ANKLE, P.R_FOOT]
        joint_idx = _PRIMARY_JOINT[primary_key][1]

    rep_end_frames = [r["end"] for r in reps]
    badge_window = max(1, int(pose.fps * 0.3))  # show each badge ~0.3s around its frame
    bar_speeds, bar_vmax = _bar_speeds(bar_xy)  # per-frame plate speed -> colour the bar path

    cap = cv2.VideoCapture(str(in_path))
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), pose.fps, (pose.width, pose.height)
    )

    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if f < pose.num_frames:
            _draw_full_skeleton(frame, lm, f)
            _draw_skeleton(frame, lm, f, chain)
            _draw_angle(frame, lm, f, joint_idx, primary)
            done = sum(1 for e in rep_end_frames if e <= f)
            cv2.putText(frame, f"Reps: {done}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, YELLOW, 2)
            _draw_badge(frame, f, rep_metrics, badge_window)
            _draw_bar_path(frame, bar_xy, f, bar_speeds, bar_vmax)
        writer.write(frame)
        f += 1

    cap.release()
    writer.release()
    return out_path


_BODY_EDGES = [
    (P.NOSE, P.L_SHOULDER), (P.NOSE, P.R_SHOULDER),
    (P.L_SHOULDER, P.R_SHOULDER), (P.L_HIP, P.R_HIP),
    (P.L_SHOULDER, P.L_HIP), (P.R_SHOULDER, P.R_HIP),
    (P.L_SHOULDER, P.L_ELBOW), (P.L_ELBOW, P.L_WRIST),
    (P.R_SHOULDER, P.R_ELBOW), (P.R_ELBOW, P.R_WRIST),
    (P.L_HIP, P.L_KNEE), (P.L_KNEE, P.L_ANKLE),
    (P.R_HIP, P.R_KNEE), (P.R_KNEE, P.R_ANKLE),
    (P.L_ANKLE, P.L_HEEL), (P.L_HEEL, P.L_FOOT), (P.L_ANKLE, P.L_FOOT),
    (P.R_ANKLE, P.R_HEEL), (P.R_HEEL, P.R_FOOT), (P.R_ANKLE, P.R_FOOT),
]
_BODY_JOINTS = sorted({s for e in _BODY_EDGES for s in e})


def _draw_full_skeleton(frame, lm, f):
    """Faint full-body skeleton from every available landmark (MediaPipe fills all 33; YOLO/RTMPose
    fill the COCO subset, so the rest are skipped). The bold analysis chain is drawn over this."""
    for a, b in _BODY_EDGES:
        pa, pb = _xy(lm, f, a), _xy(lm, f, b)
        if pa and pb:
            cv2.line(frame, pa, pb, SKELETON, 2, cv2.LINE_AA)
    for idx in _BODY_JOINTS:
        p = _xy(lm, f, idx)
        if p:
            cv2.circle(frame, p, 4, ORANGE, -1)


def _draw_skeleton(frame, lm, f, chain):
    pts = [_xy(lm, f, i) for i in chain]
    for a, b in zip(pts, pts[1:]):
        if a and b:
            cv2.line(frame, a, b, GREEN, 3)
    for p in pts:
        if p:
            cv2.circle(frame, p, 6, ORANGE, -1)


def _draw_angle(frame, lm, f, joint_idx, primary):
    jp = _xy(lm, f, joint_idx)
    if jp and not np.isnan(primary[f]):
        cv2.putText(frame, f"{primary[f]:.0f}", (jp[0] + 12, jp[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)


def _speed_color(t: float):
    """Normalized speed t in [0,1] -> BGR. Slow = blue, fast = red (no green — avoids clashing
    with the green skeleton). Unit-tested."""
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    return (int(255 * (1 - t)), 0, int(255 * t))     # blue -> red


def _bar_speeds(bar_xy):
    """Per-frame plate speed (px/frame) and a robust max (90th pct) for the colour scale."""
    if bar_xy is None:
        return None, 0.0
    n = len(bar_xy)
    speeds = np.zeros(n)
    for i in range(1, n):
        if not (np.any(np.isnan(bar_xy[i])) or np.any(np.isnan(bar_xy[i - 1]))):
            speeds[i] = float(np.hypot(bar_xy[i, 0] - bar_xy[i - 1, 0],
                                       bar_xy[i, 1] - bar_xy[i - 1, 1]))
    moving = speeds[speeds > 0]
    vmax = float(np.percentile(moving, 90)) if moving.size else 0.0
    return speeds, vmax


def _draw_bar_path(frame, bar_xy, f, speeds, vmax):
    """Bar path coloured by speed: red (slow) -> green (fast). White dot = current plate centre."""
    if bar_xy is None:
        return
    pts, spd = [], []
    for i in range(f + 1):
        if not np.any(np.isnan(bar_xy[i])):
            pts.append((int(bar_xy[i, 0]), int(bar_xy[i, 1])))
            spd.append(speeds[i] if speeds is not None else 0.0)
    for a, b, s in zip(pts, pts[1:], spd[1:]):
        t = (s / vmax) if vmax > 0 else 0.0
        cv2.line(frame, a, b, _speed_color(t), 2, cv2.LINE_AA)
    if pts:
        cv2.circle(frame, pts[-1], 6, WHITE, -1)


def _draw_badge(frame, f, rep_metrics, window):
    for rm in rep_metrics:
        if abs(f - rm["badge_frame"]) <= window:
            text, ok = rm["badge"]
            color = GREEN if ok else RED
            cv2.putText(frame, text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
            return
