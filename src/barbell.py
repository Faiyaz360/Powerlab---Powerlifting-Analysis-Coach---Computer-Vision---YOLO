"""Barbell tracking + velocity (VBT) by tracking the WEIGHT-PLATE CENTRE.

The plate is the right target (large, rigid, vivid on competition plates) — this is how Qwik/
Metric work. We segment the plate by colour (high saturation, any hue → works for blue/red/
green/yellow plates), keep the round blob nearest the bar (anchored by the wrist/shoulder from
the pose, so we don't lock onto a background plate or clothing), and take its centre + diameter
per frame. Pixels→metres comes from the 450 mm plate diameter. Velocity is a Butterworth-filtered
derivative of the plate's vertical position.

Refs: tlancon/barbellcv, kostecky/VBT-Barbell-Tracker, Balsalobre-Fernández 2021.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import pose as P

# Optional trained YOLO plate detector (colour-agnostic). Falls back to HSV if absent.
_PLATE_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "plate.pt"
_plate_model = None
_plate_model_loaded = False

PLATE_DIAMETER_M = 0.450  # IPF/IWF standard competition / bumper plate

S_MIN, V_MIN = 90, 60     # a plate is a saturated, bright colour (hue-agnostic at acquisition)
CIRCULARITY_MIN = 0.55    # contourArea / enclosing-circle area — rejects non-round blobs
HUE_TOL = 18              # once locked onto the plate's hue, reject other-colour blobs (bg plate)
MIN_PEAK_MS = 0.3         # a real concentric pull peaks above this; below = height drift, not a rep
# Lucas-Kanade optical-flow tracking of the marked plate (replaces template matching, which drifts).
_LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
_FB_ERROR_MAX = 1.5       # px — forward-backward round-trip error gate (drops points the flow lost)
_MIN_TRACK_PTS = 3        # below this the plate is treated as lost (occluded) -> hold last position
_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# (left, right) landmark slots whose midpoint sits near the plate on the camera side
_BAR_ANCHOR = {
    "deadlift": (P.L_WRIST, P.R_WRIST),
    "squat": (P.L_SHOULDER, P.R_SHOULDER),
}


def track_plate(video_path, pose: P.PoseResult, lift: str, plate_backend: str = "hsv",
                progress=None, seed=None):
    """Per-frame plate centre (x, y) in pixels and radius. Gaps interpolated.

    ``plate_backend``: "hsv" (colour+shape, default) or "yolo" (trained detector, colour-agnostic
    — needs models/plate.pt; better on same-colour-background clips but a noisier signal).
    ``seed``: optional (cx, cy, r) the user marked on frame 0 — locks the plate's colour and
    anchors the jump-gate to the right plate from the start (initialisation, not per-frame).
    Returns (centers[n,2], radii[n]).
    """
    if seed is not None:
        return _track_from_seed(video_path, pose.num_frames, seed, progress)
    anchor = _anchor_series(pose, lift)
    cap = cv2.VideoCapture(str(video_path))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = pose.num_frames
    min_r, max_r = int(h * 0.05), int(h * 0.25)
    min_area = np.pi * min_r * min_r * 0.5

    centers = np.full((n, 2), np.nan)
    radii = np.full(n, np.nan)
    model = _load_plate_model() if plate_backend == "yolo" else None  # opt-in; else HSV
    max_jump = h * 0.12          # reject frame-to-frame jumps (e.g. onto a background plate)
    prev, rejects, locked_hue = None, 0, None
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok or f >= n:
            break
        # Detect near the POSE anchor (stable, on the bar) — finds the right plate.
        if model is not None:
            hit = _detect_plate_yolo(model, frame, anchor[f], min_r, max_r)
        else:
            hit = _detect_plate(frame, anchor[f], min_r, max_r, min_area)
        if hit is not None:
            # Hue-lock only matters for the colour-based HSV detector; the trained model is
            # shape-based, so skip it there.
            hue = _patch_hue(frame, hit[0], hit[1]) if model is None else None
            jump_ok = prev is None or np.hypot(hit[0][0] - prev[0], hit[0][1] - prev[1]) <= max_jump
            hue_ok = model is not None or locked_hue is None or hue is None or _hue_diff(hue, locked_hue) <= HUE_TOL
            if jump_ok and hue_ok:
                centers[f] = hit[0]
                radii[f] = hit[1]
                prev, rejects = hit[0], 0
                if locked_hue is None and hue is not None:
                    locked_hue = hue   # lock onto the plate's colour after first good detection
            else:
                rejects += 1            # far jump or wrong colour (background plate) — drop it
                if rejects > 20:        # keep seeing it -> re-acquire position (keep hue lock)
                    prev = None
        f += 1
        if progress:
            progress(f)
    cap.release()

    centers[:, 0] = _interp(centers[:, 0])
    centers[:, 1] = _interp(centers[:, 1])
    return centers, radii


def _track_from_seed(video_path, n, seed, progress=None):
    """MANUAL bar track by LUCAS-KANADE optical flow from the plate the user marked on frame 0.

    We scatter a few strong feature points across the marked plate disc and follow each one
    frame-to-frame with pyramidal optical flow (cv2.calcOpticalFlowPyrLK). A forward-backward
    consistency check throws out points the flow lost, then the centre MOVES BY THE MEDIAN FLOW of
    the survivors (not their median position) — so it rides the plate rigidly from where the user
    marked it, and re-seeding fresh points after a floor bounce can't shift the baseline between
    reps. If too many points are lost (the plate is occluded at lockout) we hold the last centre and
    re-acquire when it reappears. This is the validated approach for barbell paths (Nagao 2022, ICC
    ~0.99 vs motion-capture) and, unlike template matching, it doesn't slide along the rim or snap
    onto the body. Radius is the marked radius (a side-on plate's on-screen size barely changes).
    The path is 1-euro smoothed.
    """
    cap = cv2.VideoCapture(str(video_path))
    scx, scy, sr = float(seed[0]), float(seed[1]), int(round(seed[2]))
    centers = np.full((n, 2), np.nan)
    radii = np.full(n, float(sr))
    prev_gray, pts, last = None, None, (scx, scy)
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok or f >= n:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if f == 0:
            pts = _seed_points(gray, scx, scy, sr)
            centers[f] = (scx, scy)
        elif pts is not None and len(pts) >= _MIN_TRACK_PTS:
            nxt, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **_LK_PARAMS)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, nxt, None, **_LK_PARAMS)
            fb = np.abs(pts - back).reshape(-1, 2).max(axis=1)      # round-trip error per point
            ok_pts = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < _FB_ERROR_MAX)
            if ok_pts.sum() >= _MIN_TRACK_PTS:
                # move the centre by the CONSENSUS motion of the survivors (median-flow), not by their
                # median position -> re-seeding after a bounce can't shift the baseline between reps
                flow = (nxt[ok_pts] - pts[ok_pts]).reshape(-1, 2)
                last = (last[0] + float(np.median(flow[:, 0])), last[1] + float(np.median(flow[:, 1])))
                centers[f] = last
                pts = nxt[ok_pts].reshape(-1, 1, 2).astype(np.float32)
                if len(pts) < 8:                                   # cloud thinning -> top it up
                    extra = _seed_points(gray, last[0], last[1], sr)
                    if extra is not None:
                        pts = np.vstack([pts, extra]).astype(np.float32)
            else:
                pts = _seed_points(gray, last[0], last[1], sr)      # lost -> hold (NaN) + re-acquire
        else:
            pts = _seed_points(gray, last[0], last[1], sr)          # too few points -> re-acquire
        prev_gray = gray
        f += 1
        if progress:
            progress(f)
    cap.release()
    cx = _one_euro(_fill_hold(centers[:, 0]))      # hold through occlusion gaps, then smooth
    cy = _one_euro(_fill_hold(centers[:, 1]))
    return np.column_stack([cx, cy]), radii


def _seed_points(gray, cx, cy, r):
    """Strong corner/edge features inside the plate disc at (cx, cy) — the points optical flow
    follows. Masked to the disc so we lock onto the plate, not the background. (N,1,2) or None."""
    mask = np.zeros(gray.shape[:2], np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), max(4, int(r * 0.85)), 255, -1)
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=40, qualityLevel=0.01,
                                  minDistance=5, mask=mask, blockSize=7)
    return pts.astype(np.float32) if pts is not None else None


def _one_euro(series, min_cutoff=1.0, beta=0.02, dt=1.0):
    """1-euro filter (Casiez 2012): smooths jitter when the bar is slow but stays responsive when
    it's fast, so it cleans the path without lagging peak velocity. ``dt`` is in frames."""
    x = np.asarray(series, dtype=float)
    if len(x) < 2:
        return x.copy()

    def _alpha(cutoff):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    out = x.copy()
    x_prev, dx_prev = x[0], 0.0
    for i in range(1, len(x)):
        dx = (x[i] - x_prev) / dt
        a_d = _alpha(1.0)
        dx_hat = a_d * dx + (1.0 - a_d) * dx_prev
        cutoff = min_cutoff + beta * abs(dx_hat)
        a = _alpha(cutoff)
        out[i] = a * x[i] + (1.0 - a) * out[i - 1]
        x_prev, dx_prev = x[i], dx_hat
    return out


def _load_plate_model():
    """Load the trained YOLO plate detector once, if present. Returns the model or None."""
    global _plate_model, _plate_model_loaded
    if not _plate_model_loaded:
        _plate_model_loaded = True
        if _PLATE_MODEL_PATH.exists():
            try:
                from ultralytics import YOLO
                _plate_model = YOLO(str(_PLATE_MODEL_PATH))
            except Exception:
                _plate_model = None
    return _plate_model


def _detect_plate_yolo(model, frame, anchor, min_r, max_r):
    """Trained-detector plate: box nearest the bar anchor (else largest). Returns ((x,y), r)."""
    import torch

    device = 0 if torch.cuda.is_available() else "cpu"
    res = model.predict(frame, device=device, verbose=False, conf=0.25)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    has_anchor = not np.any(np.isnan(anchor))
    best, best_key = None, None
    for x, y, w, h in res.boxes.xywh.cpu().numpy():
        r = (w + h) / 4.0
        if r < min_r or r > max_r:
            continue
        dist = float(np.hypot(x - anchor[0], y - anchor[1])) if has_anchor else 0.0
        key = (-dist,) if has_anchor else (w * h,)
        if best_key is None or key > best_key:
            best_key, best = key, ((float(x), float(y)), float(r))
    return best


def _detect_plate(frame, anchor, min_r, max_r, min_area):
    """Largest round, saturated blob nearest the bar anchor. Returns ((x,y), r) or None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, S_MIN, V_MIN), (180, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    has_anchor = not np.any(np.isnan(anchor))
    best, best_key = None, None
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if r < min_r or r > max_r:
            continue
        if area / (np.pi * r * r) < CIRCULARITY_MIN:
            continue
        dist = float(np.hypot(x - anchor[0], y - anchor[1])) if has_anchor else 0.0
        # nearest the anchor wins (tie-break: larger). No anchor -> largest.
        key = (-dist, area) if has_anchor else (area,)
        if best_key is None or key > best_key:
            best_key, best = key, ((x, y), r)
    return best


def _anchor_series(pose: P.PoseResult, lift: str) -> np.ndarray:
    lm = pose.landmarks
    li, ri = _BAR_ANCHOR[lift]
    n = pose.num_frames
    xy = np.full((n, 2), np.nan)
    for f in range(n):
        pts = [lm[f, i, :2] for i in (li, ri) if not np.any(np.isnan(lm[f, i, :2]))]
        if pts:
            xy[f] = np.mean(pts, axis=0)
    xy[:, 0] = _interp(xy[:, 0])
    xy[:, 1] = _interp(xy[:, 1])
    return xy


def scale_from_radii(radii: np.ndarray):
    """Metres per pixel from the median plate diameter. None if no plate ever detected."""
    valid = radii[~np.isnan(radii)]
    if len(valid) == 0:
        return None
    return PLATE_DIAMETER_M / (2.0 * float(np.median(valid)))


def scale_from_seed(radius_px):
    """Metres per pixel from a user-marked plate radius — ground truth (pure, unit-tested)."""
    if not radius_px or float(radius_px) <= 0:
        return None
    return PLATE_DIAMETER_M / (2.0 * float(radius_px))


def _eccentric_s(y, reps, i, fps, lift):
    """Lowering duration of rep i in seconds, EXCLUDING rest. Squat: the descent into this rep's
    bottom, timed from when the bar leaves the top (the rest at the top is dropped). Deadlift: the
    lowering from this rep's lockout down to the next floor touch. None if it can't be determined."""
    if lift == "deadlift":
        if i + 1 >= len(reps):
            return None
        top, nxt = reps[i]["top"], reps[i + 1]["bottom"]
        if nxt <= top:
            return None
        seg = y[top: nxt + 1]                             # lockout -> next liftoff
        yfloor, ytop = float(seg.max()), float(seg.min())
        rom = yfloor - ytop
        if rom <= 0:
            return None
        near_floor = np.where(seg >= yfloor - 0.1 * rom)[0]   # first floor touch = touchdown (rest after)
        touchdown = top + int(near_floor[0]) if len(near_floor) else nxt
        return (touchdown - top) / fps if touchdown > top else None
    start = reps[i - 1]["top"] if i > 0 else 0           # squat: from the previous lockout / start
    bottom = reps[i]["bottom"]
    if bottom <= start:
        return None
    seg = y[start: bottom + 1]                            # y is vertical px (up = smaller)
    ytop = float(seg.min())
    rom = float(seg.max()) - ytop
    if rom <= 0:
        return None
    near_top = np.where(seg <= ytop + 0.1 * rom)[0]       # frames still near the top (resting)
    descent_start = start + int(near_top[-1]) if len(near_top) else start
    return (bottom - descent_start) / fps if bottom > descent_start else None


def velocity_per_rep(bar_xy, reps, fps, scale, lift="squat"):
    """Per-rep concentric velocity from the plate's vertical motion (Butterworth-filtered).

    The concentric is the span where the bar is actually moving UP (velocity above a threshold),
    so a pause/hold at the top is excluded. Returns a list aligned with ``reps``.
    """
    y = _lowpass(_median3(_interp(bar_xy[:, 1].astype(float))), fps)
    vel_px = -np.gradient(y) * fps                 # px/s, upward positive
    out = []
    for i, r in enumerate(reps):
        valley, top = r["bottom"], r["top"]        # floor and lockout, from the BAR signal
        if top <= valley:
            out.append(None)
            continue
        # Pull start = the LAST moment near the floor before the rise to the lockout peak. This
        # anchors the concentric to the actual pull and excludes any long setup at the floor.
        seg_y = y[valley : top + 1]
        yfloor, ytop = float(seg_y.max()), float(seg_y.min())
        rom = yfloor - ytop
        if rom <= 0:
            out.append(None)
            continue
        c0 = valley + int(np.where(seg_y >= yfloor - 0.1 * rom)[0][-1])
        # rep ENDS at the FIRST moment the bar reaches the top (lockout), not the find_peaks peak —
        # so standing/resting at the top between reps isn't counted into the concentric.
        c1 = valley + int(np.where(seg_y <= ytop + 0.1 * rom)[0][0])
        if c1 <= c0:
            out.append(None)
            continue
        disp_px = float(y[c0] - y[c1])
        dt = (c1 - c0) / fps
        peak_px = float(np.max(vel_px[c0 : c1 + 1]))
        if scale and peak_px * scale < MIN_PEAK_MS:
            continue  # not a real concentric pull — drop this spurious bar rep
        mcv_px = disp_px / dt if dt > 0 else 0.0
        ecc_s = _eccentric_s(y, reps, i, fps, lift)   # lowering only, rest excluded (lift-aware)
        out.append(_pack(disp_px, mcv_px, peak_px, dt, scale, ecc_s))
    return out


def velocity_series(bar_xy, fps, scale):
    """Per-frame vertical bar velocity over the whole set (upward positive), smoothed the same way
    as the rep velocities. m/s if calibrated, else px/s. Empty-safe -> None."""
    if bar_xy is None:
        return None
    y = _lowpass(_median3(_interp(bar_xy[:, 1].astype(float))), fps)
    vel = -np.gradient(y) * fps          # px/s, upward positive
    return vel * scale if scale else vel


def _pack(disp_px, mcv_px, peak_px, dt, scale, ecc_s=None):
    calibrated = scale is not None
    return {
        "calibrated": calibrated,
        "mean_velocity_ms": round(mcv_px * scale, 3) if calibrated else None,
        "peak_velocity_ms": round(peak_px * scale, 3) if calibrated else None,
        "rom_m": round(abs(disp_px) * scale, 3) if calibrated else None,
        "mean_velocity_px_s": round(mcv_px, 1),
        "peak_velocity_px_s": round(peak_px, 1),
        "rom_px": round(abs(disp_px), 1),
        "concentric_s": round(dt, 2),
        "eccentric_s": round(ecc_s, 2) if ecc_s is not None else None,
    }


def _interp(series):
    s = series.copy()
    idx = np.arange(len(s))
    good = ~np.isnan(s)
    if good.sum() == 0:
        return s
    s[~good] = np.interp(idx[~good], idx[good], s[good])
    return s


def _fill_hold(series):
    """Carry the last good value forward across gaps (back-fill any leading gap). Holding a lost
    track in place is more honest than interpolating toward the next sighting — it keeps the plate
    at lockout instead of sliding it down before the descent (and reads zero velocity there, which
    is correct)."""
    s = series.copy()
    last = None
    for i in range(len(s)):
        if np.isnan(s[i]):
            if last is not None:
                s[i] = last
        else:
            last = s[i]
    good = np.where(~np.isnan(s))[0]
    if len(good):
        s[: good[0]] = s[good[0]]                   # back-fill any leading NaNs with the first sighting
    return s


def _patch_hue(frame, center, r):
    """Median hue of a small patch at the plate centre (for colour locking)."""
    x, y = int(center[0]), int(center[1])
    rr = max(3, int(r * 0.3))
    patch = frame[max(0, y - rr): y + rr, max(0, x - rr): x + rr]
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return float(np.median(hsv[:, :, 0]))


def _hue_diff(h1, h2):
    """Circular hue distance (OpenCV hue is 0..180)."""
    d = abs(h1 - h2)
    return min(d, 180 - d)


def detect_bar_reps(bar_xy, fps, scale, min_rom_m=0.12):
    """Reps from the PLATE's vertical motion (not the body pose).

    Each rep's concentric = floor (a height valley) -> lockout (a height peak). Peaks are found
    with a prominence of one real rep's ROM, so holds/jitter don't create spurious reps. Returns
    [{bottom, top}] frame indices.
    """
    y = _lowpass(_median3(_interp(bar_xy[:, 1].astype(float))), fps)
    height = -y  # up is positive
    try:
        from scipy.signal import find_peaks
    except Exception:
        return []
    if np.count_nonzero(np.isfinite(height)) < 2:
        return []  # plate never tracked (e.g. fully out of frame) -> no bar-based reps
    rng = float(np.nanmax(height) - np.nanmin(height))
    prom = max(min_rom_m / scale, 0.1 * rng) if scale else 0.25 * rng
    peaks, _ = find_peaks(height, prominence=prom, distance=max(1, int(fps * 0.5)))

    reps, prev = [], 0
    for pk in peaks:
        seg = height[prev:pk]
        if len(seg):
            valley = prev + int(np.argmin(seg))
            if pk > valley:
                reps.append({"bottom": valley, "top": int(pk)})
        prev = int(pk)
    return reps


def _median3(y):
    """3-tap median to kill single-frame detection jumps before filtering."""
    if len(y) < 3:
        return y
    out = y.copy()
    out[1:-1] = np.median(np.stack([y[:-2], y[1:-1], y[2:]]), axis=0)
    return out


def _lowpass(y, fps, cutoff=10.0, order=4):
    """Zero-lag Butterworth low-pass; falls back to a moving average if scipy is unavailable."""
    if len(y) <= order * 3:
        return y
    try:
        from scipy.signal import butter, filtfilt

        wn = min(cutoff / (fps / 2.0), 0.99)
        b, a = butter(order, wn, btype="low")
        return filtfilt(b, a, y)
    except Exception:
        k = 5
        return np.convolve(y, np.ones(k) / k, mode="same")
