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
PATH_FADED = (150, 150, 150)  # completed reps drawn faint grey under the bright current-rep path
START_LINE = (60, 200, 255)   # amber 'start' reference line (BGR)

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


def _body_region(lm, f):
    """Centre + max plausible radius from the stable torso points. A landmark beyond it is treated
    as a mis-detection (e.g. a limb thrown to a frame corner by occlusion) and hidden."""
    core = [lm[f, i, :2] for i in (P.L_SHOULDER, P.R_SHOULDER, P.L_HIP, P.R_HIP)
            if not np.any(np.isnan(lm[f, i, :2]))]
    if len(core) < 2:
        return None
    center = np.mean(core, axis=0)
    sp = [lm[f, i, :2] for i in (P.L_SHOULDER, P.R_SHOULDER) if not np.any(np.isnan(lm[f, i, :2]))]
    hp = [lm[f, i, :2] for i in (P.L_HIP, P.R_HIP) if not np.any(np.isnan(lm[f, i, :2]))]
    if sp and hp:
        torso = float(np.hypot(*(np.mean(sp, axis=0) - np.mean(hp, axis=0))))
    else:
        torso = float(np.max(np.ptp(np.array(core), axis=0)))
    return center, max(torso, 1.0) * 3.0


def _xy_ok(lm, f, idx, region):
    """``_xy`` but returns None when the landmark sits implausibly far from the body region."""
    p = _xy(lm, f, idx)
    if p is None or region is None:
        return p
    center, maxd = region
    return p if np.hypot(p[0] - center[0], p[1] - center[1]) <= maxd else None


def render_video(in_path, out_path, pose: P.PoseResult, analysis: dict):
    """Write an annotated mp4 to ``out_path``."""
    lm = pose.landmarks
    side = analysis["series"]["side"]
    primary_key = analysis["primary_key"]
    primary = analysis["series"][primary_key]
    reps = analysis["reps"]
    rep_metrics = analysis["rep_metrics"]
    bar_xy = analysis.get("bar_xy")
    # skeleton overlay: "side" = camera-side joints (default), "full" = all joints, "off" = bar-path only
    skeleton = analysis.get("skeleton", "side")

    if side == "left":
        chain = [P.L_SHOULDER, P.L_HIP, P.L_KNEE, P.L_ANKLE, P.L_FOOT]
        joint_idx = _PRIMARY_JOINT[primary_key][0]
    else:
        chain = [P.R_SHOULDER, P.R_HIP, P.R_KNEE, P.R_ANKLE, P.R_FOOT]
        joint_idx = _PRIMARY_JOINT[primary_key][1]

    rep_end_frames = [r["end"] for r in reps]
    badge_window = max(1, int(pose.fps * 0.3))  # show each badge ~0.3s around its frame
    bar_speeds, bar_vmax = _bar_speeds(bar_xy)  # per-frame plate speed -> colour the bar path

    # WL-style on-video overlay data: live speed HUD + a 'start' reference line at the bar's origin
    fps = pose.fps
    scale = analysis.get("scale_m_per_px")
    lift_name = analysis.get("lift", "")
    bar_load = analysis.get("bar_load")
    bar_velocity = analysis.get("bar_velocity") or []
    bar_reps = analysis.get("bar_reps") or []
    start_y = None
    if bar_xy is not None:
        valid = np.where(~np.isnan(bar_xy[:, 1]))[0]
        if len(valid):
            start_y = float(bar_xy[valid[0], 1])     # where the bar began = the start line
    rep_means = [(r["top"], bv.get("mean_velocity_ms")) for r, bv in zip(bar_reps, bar_velocity) if bv]

    # On-video real-time velocity graph: precompute the curve's pixel points once (frame dims fixed).
    vel_series = analysis.get("bar_velocity_series")
    graph_pts, graph_box = None, None
    if vel_series is not None and len(vel_series) >= 2:
        vs = np.nan_to_num(np.asarray(vel_series, dtype=float))
        vmax = float(np.max(np.abs(vs))) or 1.0
        n = len(vs)
        gx0, gx1 = int(pose.width * 0.05), int(pose.width * 0.95)
        gy0, gy1 = int(pose.height * 0.81), int(pose.height * 0.97)
        gmid = (gy0 + gy1) // 2
        xs = gx0 + (gx1 - gx0) * np.arange(n) // max(1, n - 1)
        ys = np.clip((gmid - (vs / vmax) * ((gy1 - gy0) / 2) * 0.9).astype(int), gy0, gy1)
        graph_pts = np.stack([xs, ys], axis=1).astype(np.int32)
        graph_box = (gx0, gy0, gx1, gy1, gmid, vmax)

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
            cur_start = max([e for e in rep_end_frames if e <= f], default=0)  # current-rep boundary
            if skeleton != "off":
                region = _body_region(lm, f)
                if skeleton == "full":                     # all points (front view / detailed)
                    _draw_full_skeleton(frame, lm, f, region)
                    _draw_skeleton(frame, lm, f, chain, region)
                else:                                      # "side": camera-side sagittal points only
                    _draw_side_skeleton(frame, lm, f, side, region)
                _draw_angle(frame, lm, f, joint_idx, primary)
            if start_y is not None:
                _draw_start_line(frame, start_y)
            _draw_bar_path(frame, bar_xy, f, bar_speeds, bar_vmax, cur_start)
            done = sum(1 for e in rep_end_frames if e <= f)
            speed_ms = (bar_speeds[f] * fps * scale) if (bar_speeds is not None and scale) else None
            mean_ms = next((m for tf, m in reversed(rep_means) if tf <= f), None)  # last rep's mean
            _draw_hud(frame, lift_name, bar_load, done, speed_ms, mean_ms)
            _draw_badge(frame, f, rep_metrics, badge_window)
            _draw_velocity_graph(frame, graph_pts, graph_box, f)
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

# Camera-side sagittal chain (head -> shoulder -> elbow/wrist + shoulder -> hip -> knee -> ankle -> foot).
# These are the "sideways points" that matter for a side-on squat/deadlift, without far-side clutter.
_SIDE_EDGES_L = [(P.NOSE, P.L_SHOULDER), (P.L_SHOULDER, P.L_ELBOW), (P.L_ELBOW, P.L_WRIST),
                 (P.L_SHOULDER, P.L_HIP), (P.L_HIP, P.L_KNEE), (P.L_KNEE, P.L_ANKLE),
                 (P.L_ANKLE, P.L_HEEL), (P.L_HEEL, P.L_FOOT), (P.L_ANKLE, P.L_FOOT)]
_SIDE_EDGES_R = [(P.NOSE, P.R_SHOULDER), (P.R_SHOULDER, P.R_ELBOW), (P.R_ELBOW, P.R_WRIST),
                 (P.R_SHOULDER, P.R_HIP), (P.R_HIP, P.R_KNEE), (P.R_KNEE, P.R_ANKLE),
                 (P.R_ANKLE, P.R_HEEL), (P.R_HEEL, P.R_FOOT), (P.R_ANKLE, P.R_FOOT)]


def _draw_full_skeleton(frame, lm, f, region):
    """Faint full-body skeleton from every available landmark (MediaPipe fills all 33; YOLO/RTMPose
    fill the COCO subset). Landmarks that fly far outside the body region (occlusion mis-detects)
    are hidden, not drawn way off. The bold analysis chain is drawn over this."""
    pts = {idx: _xy_ok(lm, f, idx, region) for idx in _BODY_JOINTS}
    for a, b in _BODY_EDGES:
        if pts.get(a) and pts.get(b):
            cv2.line(frame, pts[a], pts[b], SKELETON, 2, cv2.LINE_AA)
    for p in pts.values():
        if p:
            cv2.circle(frame, p, 4, ORANGE, -1)


def _draw_skeleton(frame, lm, f, chain, region):
    pts = [_xy_ok(lm, f, i, region) for i in chain]
    for a, b in zip(pts, pts[1:]):
        if a and b:
            cv2.line(frame, a, b, GREEN, 3)
    for p in pts:
        if p:
            cv2.circle(frame, p, 6, ORANGE, -1)


def _draw_side_skeleton(frame, lm, f, side, region):
    """Camera-side joints only (head, shoulder, elbow, wrist, hip, knee, ankle, heel, foot) — the
    'sideways points' for a side-on squat/deadlift, without the far-side limbs cluttering the view."""
    edges = _SIDE_EDGES_L if side == "left" else _SIDE_EDGES_R
    pts = {}
    for a, b in edges:
        pts.setdefault(a, _xy_ok(lm, f, a, region))
        pts.setdefault(b, _xy_ok(lm, f, b, region))
    for a, b in edges:
        if pts[a] and pts[b]:
            cv2.line(frame, pts[a], pts[b], GREEN, 3, cv2.LINE_AA)
    for p in pts.values():
        if p:
            cv2.circle(frame, p, 6, ORANGE, -1)


def _draw_angle(frame, lm, f, joint_idx, primary):
    jp = _xy(lm, f, joint_idx)
    if jp and not np.isnan(primary[f]):
        cv2.putText(frame, f"{primary[f]:.0f}", (jp[0] + 12, jp[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)


def _speed_color(t: float):
    """Normalised speed t in [0,1] -> BGR heatmap: slow = red, average = blue, fast = green.
    Three bands. Unit-tested."""
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    if t < 0.5:
        u = t / 0.5                                  # red -> blue
        return (int(255 * u), 0, int(255 * (1 - u)))
    u = (t - 0.5) / 0.5                              # blue -> green
    return (int(255 * (1 - u)), int(255 * u), 0)


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


def _draw_bar_path(frame, bar_xy, f, speeds, vmax, cur_start=0):
    """Bar path: completed reps faded grey, the CURRENT rep speed-coloured (blue slow -> red fast).
    ``cur_start`` is the frame the current rep began. White dot = current plate centre."""
    if bar_xy is None:
        return
    last = None
    for i in range(f + 1):
        if np.any(np.isnan(bar_xy[i])):
            continue
        p = (int(bar_xy[i, 0]), int(bar_xy[i, 1]))
        if last is not None:
            if i <= cur_start:
                cv2.line(frame, last, p, PATH_FADED, 2, cv2.LINE_AA)        # earlier reps: faint grey
            else:
                t = (speeds[i] / vmax) if (speeds is not None and vmax > 0) else 0.0
                cv2.line(frame, last, p, _speed_color(t), 5, cv2.LINE_AA)   # current rep: bright, thicker
        last = p
    if last is not None:
        cv2.circle(frame, last, 6, WHITE, -1)


def _draw_start_line(frame, start_y):
    """Amber dashed horizontal reference at the bar's starting height + a 'start' label chip
    (WL-style) — shows how far the bar drifts from where the set began."""
    w = frame.shape[1]
    y = int(start_y)
    for x in range(0, w, 28):
        cv2.line(frame, (x, y), (min(x + 16, w), y), START_LINE, 2, cv2.LINE_AA)  # amber dashes
    s = max(0.5, w / 1500.0)
    (tw, th), _ = cv2.getTextSize("start", cv2.FONT_HERSHEY_SIMPLEX, s, 2)
    cv2.rectangle(frame, (6, max(0, y - th - 12)), (6 + tw + 10, max(th + 4, y - 2)), (30, 30, 34), -1)
    cv2.putText(frame, "start", (11, max(th, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, s, START_LINE, 2, cv2.LINE_AA)


def _draw_hud(frame, lift, bar_load, rep_no, speed_ms, mean_ms):
    """Translucent top-left panel with live metrics (WL-style on-video readout): exercise, weight,
    rep count, the bar's current speed, and the last rep's mean concentric velocity. The box
    auto-sizes to its text and everything scales with the frame."""
    s = max(0.9, frame.shape[1] / 850.0)
    header = (lift or "").upper() + (f"   {bar_load:g} kg" if bar_load else "")
    rows = [(header, 0.95 * s, YELLOW), (f"REP {rep_no}", 0.8 * s, WHITE)]
    if speed_ms is not None:
        rows.append((f"{speed_ms:.2f} m/s  now", 0.8 * s, WHITE))
    if mean_ms is not None:
        rows.append((f"{mean_ms:.2f} m/s  mean", 0.8 * s, WHITE))
    thick = max(1, int(2 * s))
    pad, rh = int(16 * s), int(40 * s)
    widths = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)[0][0] for t, fs, _ in rows]
    box_w, box_h = max(widths) + pad * 2, pad * 2 + rh * len(rows)
    x0, y0 = int(18 * s), int(18 * s)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (18, 18, 22), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    y = y0 + pad + int(26 * s)
    for t, fs, col in rows:
        cv2.putText(frame, t, (x0 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, fs, col, thick, cv2.LINE_AA)
        y += rh


def _draw_velocity_graph(frame, pts, box, f):
    """Burn a compact velocity-vs-time graph onto the bottom of the frame. The curve up to the
    current frame is bright with a 'now' cursor, the rest faint — so it animates as the video plays
    (the on-video real-time graph). Points are precomputed; drawing is O(n) via polylines."""
    if pts is None or box is None:
        return
    gx0, gy0, gx1, gy1, gmid, vmax = box
    overlay = frame.copy()
    cv2.rectangle(overlay, (gx0 - 8, gy0 - 8), (gx1 + 8, gy1 + 8), (18, 18, 22), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.line(frame, (gx0, gmid), (gx1, gmid), (90, 90, 96), 1, cv2.LINE_AA)   # zero baseline
    cv2.polylines(frame, [pts], False, (115, 115, 120), 1, cv2.LINE_AA)       # full curve, faint
    k = min(f + 1, len(pts))
    if k >= 2:
        cv2.polylines(frame, [pts[:k]], False, START_LINE, 2, cv2.LINE_AA)    # past curve, bright
    cx = int(pts[min(f, len(pts) - 1)][0])
    cv2.line(frame, (cx, gy0), (cx, gy1), WHITE, 1, cv2.LINE_AA)              # 'now' cursor
    # label + slow/avg/fast colour key (drawn last so they stay readable over the curve)
    s = max(0.5, (gx1 - gx0) / 1100.0)
    cv2.putText(frame, f"BAR SPEED (peak {vmax:.2f} m/s)", (gx0 + 4, gy0 + int(18 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (215, 215, 219), 1, cv2.LINE_AA)
    lx = gx1 - int(168 * s)
    for label, frac in (("slow", 0.0), ("avg", 0.5), ("fast", 1.0)):
        cv2.circle(frame, (lx, gy0 + int(13 * s)), max(2, int(5 * s)), _speed_color(frac), -1)
        cv2.putText(frame, label, (lx + int(9 * s), gy0 + int(18 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, (215, 215, 219), 1, cv2.LINE_AA)
        lx += int(56 * s)


def _draw_badge(frame, f, rep_metrics, window):
    for rm in rep_metrics:
        if abs(f - rm["badge_frame"]) <= window:
            text, ok = rm["badge"]
            color = GREEN if ok else RED
            cv2.putText(frame, text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
            return
