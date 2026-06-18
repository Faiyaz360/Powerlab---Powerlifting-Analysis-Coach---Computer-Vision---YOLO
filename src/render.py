"""Draw the analysis onto the video: side skeleton, primary joint angle, rep counter, badge.

Lift-agnostic: it reads ``analysis['primary_key']`` (knee for squat, hip for deadlift) and the
per-rep ``badge`` / ``badge_frame``, so squat and deadlift share the same renderer.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import advanced_metrics as am
from . import angles
from . import charts
from . import pose as P

GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
MAGENTA = (255, 0, 255)
SKEL_LINE = (0, 255, 255)   # bold analysis-chain colour = TRUE yellow (BGR); clear of the bar-path
#                             red->blue->green gradient so the skeleton never blends into the bar line
JOINT = (255, 255, 255)     # joint dots = white (also clear of the bar-path colours)
SKELETON = (230, 230, 230)  # faint full-body skeleton, under the bold analysis chain
PATH_FADED = (150, 150, 150)  # completed reps drawn faint grey under the bright current-rep path
START_LINE = (60, 200, 255)   # amber 'start' reference line (BGR)
SPINE = (255, 100, 210)       # bright violet back-axis line (BGR) — distinct from skeleton/bar-path


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


def _start_anchor_frame(bar_xy, bar_reps, lift, valid):
    """Frame the 'start' reference lines (amber height + white centre) anchor to: the literal MOMENT
    THE LIFT STARTS in rep 1 — the squat begins descending, the deadlift breaks the floor — not the
    video start (mid-walkout/setup, which pins a squat's centre line to the rack and reads fake
    drift). Deadlift: the first floor valley (bar about to rise). Squat: the last frame still near the
    standing top before the first descent (the bar about to drop). Falls back to the first tracked
    frame when there are no reps. ``bar_xy[:,1]`` is vertical px, down = larger. Pure — unit-tested."""
    if not bar_reps:
        return int(valid[0])
    b = int(bar_reps[0]["bottom"])
    if lift == "deadlift":
        return b if 0 <= b < len(bar_xy) else int(valid[0])   # floor valley = the bar breaks the floor
    y = bar_xy[:b + 1, 1].astype(float)                       # squat: video start -> first deep bottom
    if y.size == 0 or np.all(np.isnan(y)):
        return int(valid[0])
    ytop = float(np.nanmin(y))                                # standing = highest bar = smallest y
    rom = float(np.nanmax(y)) - ytop
    near_top = np.where(y <= ytop + 0.1 * max(rom, 1.0))[0]   # frames still near the standing top
    return int(near_top[-1]) if len(near_top) else int(valid[0])   # last one = the descent start


def render_video(in_path, out_path, pose: P.PoseResult, analysis: dict):
    """Write an annotated mp4 to ``out_path``."""
    lm = pose.landmarks
    side = analysis["series"]["side"]
    reps = analysis["reps"]
    rep_metrics = analysis["rep_metrics"]
    bar_xy = analysis.get("bar_xy")
    # skeleton overlay: "side" = camera-side joints (default), "full" = all joints, "off" = bar-path only
    skeleton = analysis.get("skeleton", "side")
    lean_series = analysis["series"].get("lean")   # per-frame torso lean -> the back-axis tracker angle

    if side == "left":
        chain = [P.L_SHOULDER, P.L_HIP, P.L_KNEE, P.L_ANKLE, P.L_FOOT]
    else:
        chain = [P.R_SHOULDER, P.R_HIP, P.R_KNEE, P.R_ANKLE, P.R_FOOT]

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
    start_y = start_x = None
    if bar_xy is not None:
        valid = np.where(~np.isnan(bar_xy[:, 1]))[0]
        if len(valid):
            f0 = _start_anchor_frame(bar_xy, bar_reps, lift_name, valid)  # first rep, not video start
            start_y = float(bar_xy[f0, 1])           # bar's height at rep 1 = the 'start' line
            start_x = float(bar_xy[f0, 0])           # bar's X at rep 1 = the vertical 'centre' line
    rep_means = [(r["top"], bv.get("mean_velocity_ms")) for r, bv in zip(bar_reps, bar_velocity) if bv]
    # Real-time bar-path panel (top-right): 2D trajectory, concentric (up) green / eccentric (down) blue.
    path_panel_pts = path_runs = path_panel_box = None
    if bar_xy is not None and start_x is not None:
        x = bar_xy[:, 0].astype(float)
        y = bar_xy[:, 1].astype(float)
        pw = max(150, int(pose.width * 0.34))
        ph = int(pw * 1.05)
        bx0, by0, cxp = pose.width - pw - 12, 12, pose.width - pw - 12 + pw // 2
        dxr = float(np.nanmax(np.abs(x - start_x))) or 1.0
        ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
        yr = (ymax - ymin) or 1.0
        ytop, ybot = by0 + int(ph * 0.20), by0 + int(ph * 0.84)
        panel_x = np.clip(cxp + ((x - start_x) / dxr) * (pw * 0.40), bx0 + 4, bx0 + pw - 4)
        panel_y = ytop + ((y - ymin) / yr) * (ybot - ytop)
        path_panel_pts = np.stack([panel_x, panel_y], axis=1).astype(np.int32)
        up = np.zeros(len(y), bool)
        up[1:] = (y[1:] - y[:-1]) < 0                 # moving up = concentric
        path_runs, s0 = [], 1                         # group frames into same-direction runs
        for i in range(2, len(y)):
            if up[i] != up[i - 1]:
                path_runs.append((s0, i - 1, bool(up[i - 1])))
                s0 = i
        if len(y) > 1:
            path_runs.append((s0, len(y) - 1, bool(up[-1])))
        path_panel_box = (bx0, by0, bx0 + pw, by0 + ph, cxp, ytop, ybot)

    # Compact rep table (Rep | Vel | OK) stacked UNDER the bar-path panel, top-right.
    made_flags = [rm.get("depth_pass", rm.get("lockout_pass")) for rm in rep_metrics]  # ✓/✗ per rep
    table_img = table_xy = None
    if path_panel_box is not None:
        table_img = charts.velocity_table_img(bar_velocity, path_panel_box[2] - path_panel_box[0],
                                              made_flags)
        if table_img is not None:
            table_xy = (path_panel_box[0], path_panel_box[3] + 8)   # just below the panel

    # On-video real-time velocity graph: full-width strip along the bottom.
    vel_series = analysis.get("bar_velocity_series")
    graph_pts, graph_box, graph_reps = None, None, None
    badge_y = int(pose.height * 0.80)             # the depth flash sits just above the graph
    if vel_series is not None and len(vel_series) >= 2:
        vs = np.nan_to_num(np.asarray(vel_series, dtype=float))
        vmax = float(np.max(np.abs(vs))) or 1.0
        n = len(vs)
        gx0, gx1 = int(pose.width * 0.05), int(pose.width * 0.95)
        gy1 = int(pose.height * 0.97)
        gy0 = gy1 - int(pose.height * 0.14)
        gmid = (gy0 + gy1) // 2
        xs = gx0 + (gx1 - gx0) * np.arange(n) // max(1, n - 1)
        ys = np.clip((gmid - (vs / vmax) * ((gy1 - gy0) / 2) * 0.9).astype(int), gy0, gy1)
        graph_pts = np.stack([xs, ys], axis=1).astype(np.int32)
        graph_box = (gx0, gy0, gx1, gy1, gmid, vmax)
        badge_y = gy0
        # Each rep's concentric span (LIFTOFF -> lockout) as series indices: the green fill area and
        # the red 'rep start' marker. Liftoff = the bar breaking the floor (not the rest valley, which
        # can sit seconds back). Frame index == series index == graph-point index.
        graph_reps = [(int(r.get("liftoff", r["bottom"])), min(int(r["top"]), n - 1)) for r in bar_reps
                      if 0 <= int(r.get("liftoff", r["bottom"])) < n] if bar_reps else None

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
            region = _body_region(lm, f)
            if skeleton != "off":
                if skeleton == "full":                     # all points (front view / detailed)
                    _draw_full_skeleton(frame, lm, f, region)
                    _draw_skeleton(frame, lm, f, chain, region)
                else:                                      # "side": camera-side sagittal points only
                    _draw_side_skeleton(frame, lm, f, side, region)
                _draw_joint_angles(frame, lm, f, side, region)
            # Back-axis tracker (always on): the torso-lean line + live angle. HONEST SCOPE — two real
            # landmarks, so it shows back LEAN / hinge, not spinal curvature (that needs 3D, Phase 3).
            lean_f = float(lean_series[f]) if (lean_series is not None and f < len(lean_series)) else None
            _draw_back_line(frame, lm, f, side, region, lean_f)
            if start_y is not None:
                _draw_start_line(frame, start_y)
            if start_x is not None:
                cur = bar_xy[f] if (bar_xy is not None and f < len(bar_xy)
                                    and not np.any(np.isnan(bar_xy[f]))) else None
                _draw_center_line(frame, start_x, cur, scale)
            _draw_bar_path(frame, bar_xy, f, bar_speeds, bar_vmax, cur_start)
            done = sum(1 for e in rep_end_frames if e <= f)
            speed_ms = (bar_speeds[f] * fps * scale) if (bar_speeds is not None and scale) else None
            mean_ms = next((m for tf, m in reversed(rep_means) if tf <= f), None)  # last rep's mean
            _draw_hud(frame, lift_name, bar_load, done, speed_ms, mean_ms,
                      name=analysis.get("lifter_name"), sex=analysis.get("sex"),
                      bodyweight=analysis.get("bodyweight"), lift_score=analysis.get("lift_score"))
            _draw_badge(frame, f, rep_metrics, badge_window, badge_y)
            _draw_velocity_graph(frame, graph_pts, graph_box, f, graph_reps)
            _draw_path_panel(frame, path_panel_pts, path_runs, path_panel_box, f)
            if table_img is not None and table_xy is not None:
                th, tw = table_img.shape[:2]
                tx, ty = table_xy
                if 0 <= tx and ty + th <= frame.shape[0] and tx + tw <= frame.shape[1]:
                    frame[ty:ty + th, tx:tx + tw] = table_img
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
            cv2.circle(frame, p, 4, JOINT, -1)


def _draw_skeleton(frame, lm, f, chain, region):
    pts = [_xy_ok(lm, f, i, region) for i in chain]
    for a, b in zip(pts, pts[1:]):
        if a and b:
            cv2.line(frame, a, b, SKEL_LINE, 3)
    for p in pts:
        if p:
            cv2.circle(frame, p, 6, JOINT, -1)


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
            cv2.line(frame, pts[a], pts[b], SKEL_LINE, 3, cv2.LINE_AA)
    for p in pts.values():
        if p:
            cv2.circle(frame, p, 6, JOINT, -1)


def _draw_joint_angles(frame, lm, f, side, region):
    """Live joint-angle numbers at the camera-side KNEE and HIP — the joints that matter for squat
    depth and deadlift hinge. Computed per frame from the landmarks; occluded joints are skipped."""
    if side == "left":
        joints = [(P.L_HIP, P.L_KNEE, P.L_ANKLE), (P.L_SHOULDER, P.L_HIP, P.L_KNEE)]
    else:
        joints = [(P.R_HIP, P.R_KNEE, P.R_ANKLE), (P.R_SHOULDER, P.R_HIP, P.R_KNEE)]
    s = max(0.5, frame.shape[1] / 1300.0)
    for a, b, c in joints:
        pa, pb, pc = _xy_ok(lm, f, a, region), _xy_ok(lm, f, b, region), _xy_ok(lm, f, c, region)
        if pa and pb and pc:
            ang = angles.calc_angle(pa, pb, pc)
            if not np.isnan(ang):
                cv2.putText(frame, f"{ang:.0f}", (pb[0] + int(10 * s), pb[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, WHITE, max(1, int(2 * s)), cv2.LINE_AA)


def _draw_back_line(frame, lm, f, side, region, lean_deg):
    """Bold 'back axis' line: low-back (camera-side hip) -> upper-back (camera-side shoulder), plus the
    live torso-lean angle. HONEST SCOPE: built from 2 points, so it shows back LEAN / hinge angle, not
    spinal ROUNDING — true back curvature needs a 3D spine (Phase 3); the 2D silhouette attempt was
    removed as too approximate to be honest."""
    hip = _xy_ok(lm, f, P.L_HIP if side == "left" else P.R_HIP, region)
    sh = _xy_ok(lm, f, P.L_SHOULDER if side == "left" else P.R_SHOULDER, region)
    if not hip or not sh:
        return
    s = max(0.5, frame.shape[1] / 1300.0)
    dx, dy = sh[0] - hip[0], sh[1] - hip[1]
    p_low = (int(hip[0] - 0.08 * dx), int(hip[1] - 0.08 * dy))   # extend ~8% past each end so the back
    p_up = (int(sh[0] + 0.08 * dx), int(sh[1] + 0.08 * dy))      # axis reads as a line, hugging the back
    cv2.line(frame, p_low, p_up, SPINE, max(2, int(4 * s)), cv2.LINE_AA)
    cv2.circle(frame, hip, max(3, int(5 * s)), SPINE, -1)
    cv2.circle(frame, sh, max(3, int(5 * s)), SPINE, -1)
    if lean_deg is not None and not np.isnan(lean_deg):
        mx, my = (hip[0] + sh[0]) // 2, (hip[1] + sh[1]) // 2
        cv2.putText(frame, f"back {lean_deg:.0f}", (mx + int(12 * s), my),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, SPINE, max(1, int(2 * s)), cv2.LINE_AA)


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


def _draw_center_line(frame, start_x, cur_xy, scale):
    """Vertical dashed reference at the bar's starting X (the 'centre' line) + the live sideways
    drift in cm — LIFT-APP style, so you see how far the bar wanders from where it began."""
    h = frame.shape[0]
    sx = int(start_x)
    for y in range(0, h, 26):
        cv2.line(frame, (sx, y), (sx, min(y + 13, h)), (170, 170, 174), 1, cv2.LINE_AA)
    if cur_xy is not None and scale:
        drift_cm = (float(cur_xy[0]) - start_x) * scale * 100
        s = max(0.5, frame.shape[1] / 1300.0)
        cv2.putText(frame, f"drift {drift_cm:+.1f} cm", (sx + 6, int(34 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (170, 170, 174), 1, cv2.LINE_AA)


_GRADE_COLORS = {                       # tier colour (BGR)
    "S": (80, 205, 255),                # gold
    "A+": (95, 205, 95), "A": (95, 205, 95),    # green
    "B": (235, 165, 70),                # blue
    "C": (70, 180, 245),                # orange
    "D": (80, 80, 235), "E": (80, 80, 235),     # red
}


def _draw_score_badge(frame, lift_score, x, y, s):
    """Gamified /100 lift-score badge under the HUD: a tier-coloured grade chip (S / A+ / A / ...)
    next to the big score number. Tier colour: S gold, A green, B blue, C orange, D-E red."""
    if not lift_score or lift_score.get("score") is None:
        return
    f = cv2.FONT_HERSHEY_SIMPLEX
    grade = str(lift_score.get("grade") or "")
    val = lift_score["score"]
    color = _GRADE_COLORS.get(grade, (200, 200, 200))
    chip = int(48 * s)
    pill_w = chip + int(152 * s)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + pill_w, y + chip), (14, 14, 18), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
    cv2.rectangle(frame, (x, y), (x + pill_w, y + chip), color, max(1, int(1.5 * s)))   # tier border
    cv2.rectangle(frame, (x, y), (x + chip, y + chip), color, -1)                        # grade chip
    (gw, gh), _ = cv2.getTextSize(grade, f, 0.9 * s, max(2, int(2 * s)))
    cv2.putText(frame, grade, (x + (chip - gw) // 2, y + (chip + gh) // 2), f, 0.9 * s,
                (20, 20, 24), max(2, int(2 * s)), cv2.LINE_AA)
    tx = x + chip + int(12 * s)
    cv2.putText(frame, "LIFT SCORE", (tx, y + int(15 * s)), f, 0.4 * s, (165, 165, 172), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{val}", (tx, y + int(40 * s)), f, 1.0 * s, WHITE, max(2, int(2 * s)), cv2.LINE_AA)
    (nw, _), _ = cv2.getTextSize(f"{val}", f, 1.0 * s, max(2, int(2 * s)))
    cv2.putText(frame, "/100", (tx + nw + int(5 * s), y + int(40 * s)), f, 0.5 * s,
                (170, 170, 176), 1, cv2.LINE_AA)


def _draw_hud(frame, lift, bar_load, rep_no, speed_ms, mean_ms,
              name=None, sex=None, bodyweight=None, lift_score=None):
    """Translucent top-left panel: a compact lifter line (name · gender · bodyweight), then exercise/
    weight, rep, live + mean bar speed and RPE — with a gamified /100 score badge under it. The box
    auto-sizes to its text and everything scales with the frame."""
    s = max(0.72, frame.shape[1] / 1050.0)        # slightly smaller HUD
    lifter = "  |  ".join(p for p in (       # ASCII '|' — cv2's Hershey font can't render a '·'
        ((name or "").strip().upper()[:16] or None),
        {"male": "M", "female": "F"}.get(sex),
        (f"{bodyweight:g} kg BW" if bodyweight else None),
    ) if p)
    rows = []
    if lifter:
        rows.append((lifter, 0.6 * s, (175, 175, 180)))     # small + muted: present, uncluttered
    header = (lift or "").upper() + (f"   {bar_load:g} kg" if bar_load else "")
    rows += [(header, 0.95 * s, YELLOW), (f"REP {rep_no}", 0.8 * s, WHITE)]
    if speed_ms is not None:
        rows.append((f"{speed_ms:.2f} m/s  now", 0.8 * s, WHITE))
    if mean_ms is not None:
        rows.append((f"{mean_ms:.2f} m/s  mean", 0.8 * s, WHITE))
        rpe = am.velocity_to_rpe(mean_ms, lift)
        if rpe is not None:
            rows.append((f"RPE ~{rpe:.1f}", 0.8 * s, (120, 220, 120)))
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
    _draw_score_badge(frame, lift_score, x0, y0 + box_h + int(8 * s), s)   # gamified score under the HUD


def _draw_velocity_graph(frame, pts, box, f, reps_idx=None):
    """Burn a compact velocity-vs-time graph onto the bottom of the frame. The curve up to the
    current frame is bright with a 'now' cursor, the rest faint. Each detected rep's concentric (the
    bar driving up) is filled green and its start marked with a red dotted vertical — both revealed
    as the bar reaches them, so the rep structure animates with the lift. ``reps_idx`` is a list of
    (start, end) series indices per rep. Points are precomputed; drawing is O(n) via polylines."""
    if pts is None or box is None:
        return
    gx0, gy0, gx1, gy1, gmid, vmax = box
    overlay = frame.copy()
    cv2.rectangle(overlay, (gx0 - 8, gy0 - 8), (gx1 + 8, gy1 + 8), (18, 18, 22), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.line(frame, (gx0, gmid), (gx1, gmid), (90, 90, 96), 1, cv2.LINE_AA)   # zero baseline
    # green fill of each rep's concentric (above-baseline) area, revealed up to the current frame
    if reps_idx:
        fill = frame.copy()
        drew = False
        for b, t in reps_idx:
            if b > f:
                break
            e = min(t, f, len(pts) - 1)
            if e <= b:
                continue
            seg = pts[b:e + 1]
            edge = [(int(x), min(int(y), gmid)) for x, y in seg]   # clamp to the up (positive) band
            poly = np.array(edge + [(int(seg[-1][0]), gmid), (int(seg[0][0]), gmid)], np.int32)
            cv2.fillPoly(fill, [poly], _CONCENTRIC)
            drew = True
        if drew:
            cv2.addWeighted(fill, 0.30, frame, 0.70, 0, frame)
    cv2.polylines(frame, [pts], False, (115, 115, 120), 1, cv2.LINE_AA)       # full curve, faint
    k = min(f + 1, len(pts))
    if k >= 2:
        cv2.polylines(frame, [pts[:k]], False, START_LINE, 2, cv2.LINE_AA)    # past curve, bright
    # red dotted 'rep start' verticals — one per rep, appearing as the bar reaches each
    if reps_idx:
        for b, _t in reps_idx:
            if b > f:
                break
            rx = int(pts[min(b, len(pts) - 1)][0])
            for yy in range(gy0, gy1, 10):
                cv2.line(frame, (rx, yy), (rx, min(yy + 5, gy1)), RED, 1, cv2.LINE_AA)
    cx = int(pts[min(f, len(pts) - 1)][0])
    cv2.line(frame, (cx, gy0), (cx, gy1), WHITE, 1, cv2.LINE_AA)              # 'now' cursor
    # label + slow/avg/fast colour key (drawn last so they stay readable over the curve)
    s = max(0.5, (gx1 - gx0) / 1100.0)
    cv2.putText(frame, f"BAR SPEED (peak {vmax:.2f} m/s)", (gx0 + 4, gy0 + int(18 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (215, 215, 219), 1, cv2.LINE_AA)
    ry = gy0 + int(34 * s)                                   # red 'rep start' key under the title
    for x in range(gx0 + 4, gx0 + 4 + int(18 * s), 5):
        cv2.line(frame, (x, ry), (min(x + 3, gx0 + 4 + int(18 * s)), ry), RED, 1, cv2.LINE_AA)
    cv2.putText(frame, "rep start", (gx0 + 4 + int(24 * s), ry + int(4 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42 * s, (215, 215, 219), 1, cv2.LINE_AA)
    lx = gx1 - int(168 * s)
    for label, frac in (("slow", 0.0), ("avg", 0.5), ("fast", 1.0)):
        cv2.circle(frame, (lx, gy0 + int(13 * s)), max(2, int(5 * s)), _speed_color(frac), -1)
        cv2.putText(frame, label, (lx + int(9 * s), gy0 + int(18 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, (215, 215, 219), 1, cv2.LINE_AA)
        lx += int(56 * s)


_CONCENTRIC = (90, 210, 90)   # green (BGR)
_ECCENTRIC = (235, 140, 40)   # blue (BGR)


def _draw_path_panel(frame, pts, runs, box, f):
    """Real-time bar-path panel (top-right): the bar's 2D trajectory drawn up to the current frame,
    concentric (up) green and eccentric (down) blue, with a 'now' dot and a con/ecc key."""
    if pts is None or box is None:
        return
    bx0, by0, bx1, by1, cxp, ytop, ybot = box
    overlay = frame.copy()
    cv2.rectangle(overlay, (bx0, by0), (bx1, by1), (22, 24, 28), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.line(frame, (cxp, ytop), (cxp, ybot), (110, 110, 116), 1, cv2.LINE_AA)   # centre (start X)
    s = max(0.4, (bx1 - bx0) / 360.0)
    cv2.putText(frame, "bar path", (bx0 + 6, by0 + int(18 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5 * s, (220, 220, 224), 1, cv2.LINE_AA)
    for a, b, is_up in runs:
        if a > f:
            break
        e = min(b, f)
        if e > a:
            cv2.polylines(frame, [pts[a:e + 1]], False, _CONCENTRIC if is_up else _ECCENTRIC,
                          2, cv2.LINE_AA)
    cf = min(f, len(pts) - 1)
    cv2.circle(frame, (int(pts[cf][0]), int(pts[cf][1])), 3, WHITE, -1)          # 'now' dot
    ly = by1 - int(10 * s)
    cv2.circle(frame, (bx0 + 12, ly), 4, _CONCENTRIC, -1)
    cv2.putText(frame, "con", (bx0 + 20, ly + int(4 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.42 * s,
                (205, 205, 209), 1, cv2.LINE_AA)
    cv2.circle(frame, (bx0 + int(70 * s), ly), 4, _ECCENTRIC, -1)
    cv2.putText(frame, "ecc", (bx0 + int(78 * s), ly + int(4 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                0.42 * s, (205, 205, 209), 1, cv2.LINE_AA)


def _draw_badge(frame, f, rep_metrics, window, badge_y):
    """When a rep hits depth (squat) / lockout (deadlift), flash a bold badge on the right just above
    the velocity graph — green when made, red when not."""
    for rm in rep_metrics:
        if abs(f - rm["badge_frame"]) <= window:
            text, ok = rm["badge"]
            label = text.replace(" OK", "") if ok else text   # cv2 can't draw unicode ticks
            color = GREEN if ok else RED
            s = max(0.8, frame.shape[1] / 800.0)
            thick = max(2, int(3 * s))
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1 * s, thick)
            x = int(frame.shape[1] * 0.95) - tw            # right-aligned
            y = badge_y - int(12 * s)                      # just above the graph
            cv2.rectangle(frame, (x - 12, y - th - 12), (x + tw + 12, y + 12), (20, 20, 24), -1)
            cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1 * s, color, thick, cv2.LINE_AA)
            return
