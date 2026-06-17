"""Turn raw landmarks into lift metrics.

Shared per-frame signals live in ``compute_series``. Each lift has its own analyzer
(``analyze_squat`` / ``analyze_deadlift``) that returns the SAME shape so render/report/faults
can stay mostly generic:

    {series, reps, rep_metrics, fps, rep_count, lift, primary_key}

``primary_key`` names the series used for rep detection + the on-frame angle ("knee" for
squat, "hip" for deadlift). Each entry in ``rep_metrics`` carries ``badge_frame`` and
``badge=(text, ok)`` so the renderer can draw a pass/fail badge without knowing the lift.

Thresholds are named constants — tune them to the individual lifter after seeing real video.
"""
from __future__ import annotations

import numpy as np

from . import pose as P
from .angles import angle_from_vertical, calc_angle

# --- squat tuning ---
STAND_KNEE_ANGLE = 160.0      # knee angle above this = standing tall (between reps)
MIN_SQUAT_KNEE_ANGLE = 140.0  # a rep's bottom must dip below this to count
DEPTH_KNEE_MAX = 100.0        # knee angle at bottom <= this ~ at/below parallel (proxy; tune per lifter)
MIN_SQUAT_RANGE = 40.0        # knee angle must swing at least this much for a real rep

# --- deadlift tuning ---
DL_STAND_HIP = 150.0          # hip angle above this = standing-ish (between reps)
DL_BOTTOM_HIP = 120.0         # a pull's bottom must hinge below this to count
# Lockout per IPF 2026 §4.3.1: stand erect (#2 — the front deltoid ends up behind the bar's vertical
# projection; we proxy "erect" by the TORSO being ~vertical at the top, i.e. the shoulder lined up
# over the hip) AND knees locked straight (#3). Torso lean from vertical has an absolute, camera-
# robust scale (0 = upright), so both use absolute thresholds — measured lockouts read ~4-6° lean /
# ~162-171° knee on the eval clip. (The exact "delt behind the BAR" check, using the tracked plate x
# + the lifter's facing, is a later refinement.)
ERECT_LEAN_MAX = 15.0         # torso tilt from vertical at the top <= this = standing erect
KNEE_LOCK_MIN = 150.0         # knee angle at the top >= this = knees locked straight (lenient floor)
HIP_RISE_FRAC = 0.33          # fraction of the ascent counted as "early pull"
# rep-sanity filters (drop detector artifacts)
MIN_REP_S = 0.8               # reps shorter than this are noise blips
MIN_REP_RANGE = 40.0          # hip angle must rise at least this much from bottom to top
MAX_ASCENT_S = 8.0            # ascent longer than this = setup merged into the rep, drop it

SMOOTH_WINDOW = 5             # moving-average window (frames) to de-jitter angle signals


# ---------------------------------------------------------------- shared series

def _pick_side(landmarks: np.ndarray) -> str:
    """Pick the body side facing the camera, by mean landmark visibility."""

    def vis(idxs):
        col = landmarks[:, idxs, 2]
        if not np.any(np.isfinite(col)):
            return 0.0
        return float(np.nanmean(col))

    left = vis([P.L_SHOULDER, P.L_HIP, P.L_KNEE, P.L_ANKLE])
    right = vis([P.R_SHOULDER, P.R_HIP, P.R_KNEE, P.R_ANKLE])
    return "left" if left >= right else "right"


def _interp_nan(series: np.ndarray) -> np.ndarray:
    s = series.astype(float).copy()
    idx = np.arange(len(s))
    good = ~np.isnan(s)
    if good.sum() == 0:
        return s
    s[~good] = np.interp(idx[~good], idx[good], s[good])
    return s


def _smooth(series: np.ndarray, k: int = SMOOTH_WINDOW) -> np.ndarray:
    if len(series) < k or k <= 1:
        return series
    return np.convolve(series, np.ones(k) / k, mode="same")


def compute_series(pose: P.PoseResult) -> dict:
    """Per-frame signals used by every lift analyzer."""
    lm = pose.landmarks
    side = _pick_side(lm)
    if side == "left":
        sh, hip, kn, an, wr = P.L_SHOULDER, P.L_HIP, P.L_KNEE, P.L_ANKLE, P.L_WRIST
    else:
        sh, hip, kn, an, wr = P.R_SHOULDER, P.R_HIP, P.R_KNEE, P.R_ANKLE, P.R_WRIST

    n = pose.num_frames
    knee = np.full(n, np.nan)
    hipang = np.full(n, np.nan)
    lean = np.full(n, np.nan)
    hip_y = np.full(n, np.nan)
    knee_y = np.full(n, np.nan)
    shoulder_y = np.full(n, np.nan)
    wrist = np.full((n, 2), np.nan)

    for f in range(n):
        S, H, K, A = lm[f, sh, :2], lm[f, hip, :2], lm[f, kn, :2], lm[f, an, :2]
        knee[f] = calc_angle(H, K, A)        # hip-knee-ankle
        hipang[f] = calc_angle(S, H, K)      # shoulder-hip-knee
        lean[f] = angle_from_vertical(S, H)  # torso tilt from vertical
        hip_y[f] = H[1]
        knee_y[f] = K[1]
        shoulder_y[f] = S[1]
        wrist[f] = lm[f, wr, :2]

    return {
        "side": side,
        "knee": _smooth(_interp_nan(knee)),
        "hip": _smooth(_interp_nan(hipang)),
        "lean": _smooth(_interp_nan(lean)),
        "hip_y": _interp_nan(hip_y),
        "knee_y": _interp_nan(knee_y),
        "shoulder_y": _interp_nan(shoulder_y),
        "wrist": wrist,  # bar-path proxy until real plate tracking (Phase 4)
    }


# ---------------------------------------------------------------- rep detection

def detect_reps(signal: np.ndarray, stand: float, min_bottom: float):
    """Find reps as excursions below ``stand`` that dip past ``min_bottom``.

    Generic over the driving signal (knee angle for squat, hip angle for deadlift).
    Returns a list of {start, bottom, end} frame indices.
    """
    reps = []
    n = len(signal)
    in_rep = False
    start = 0
    for f in range(n):
        a = signal[f]
        if np.isnan(a):
            continue
        if not in_rep and a < stand:
            in_rep = True
            start = f
        elif in_rep and a >= stand:
            _close_rep(reps, signal, start, f, min_bottom)
            in_rep = False
    if in_rep:
        _close_rep(reps, signal, start, n - 1, min_bottom)
    return reps


def _close_rep(reps, signal, start, end, min_bottom):
    seg = signal[start : end + 1]
    if len(seg) and np.nanmin(seg) < min_bottom:
        bottom = start + int(np.nanargmin(seg))
        reps.append({"start": start, "bottom": bottom, "end": end})


# ---------------------------------------------------------------- dispatch

def analyze(pose: P.PoseResult, lift: str) -> dict:
    if lift == "squat":
        return analyze_squat(pose)
    if lift == "deadlift":
        return analyze_deadlift(pose)
    raise NotImplementedError(f"No analyzer for lift '{lift}' yet.")


# ---------------------------------------------------------------- squat

def analyze_squat(pose: P.PoseResult) -> dict:
    s = compute_series(pose)
    candidates = detect_reps(s["knee"], STAND_KNEE_ANGLE, MIN_SQUAT_KNEE_ANGLE)
    fps = pose.fps

    kept_reps = []
    rep_metrics = []
    for r in candidates:
        start, bottom, end = r["start"], r["bottom"], r["end"]
        seg = s["knee"][start : end + 1]
        knee_range = float(np.nanmax(seg) - np.nanmin(seg)) if len(seg) else 0.0
        ascent_s = (end - bottom) / fps
        descent_s = (bottom - start) / fps
        duration_s = (end - start) / fps
        if _is_spurious_squat(end, bottom, ascent_s, descent_s, duration_s, knee_range):
            continue

        min_knee = float(s["knee"][bottom])
        # Depth = hip crease below the top of the knee. The geometric check (hip_y >= knee_y) is
        # the truth but needs clean landmarks; the knee-angle proxy rescues deep reps when the
        # hip landmark is misplaced (occlusion / off-axis camera). Pass if EITHER says deep.
        depth_pass = bool(min_knee <= DEPTH_KNEE_MAX or s["hip_y"][bottom] >= s["knee_y"][bottom])
        kept_reps.append({"start": start, "bottom": bottom, "end": end})
        rep_metrics.append(
            {
                "start_s": round(start / fps, 2),
                "bottom_s": round(bottom / fps, 2),
                "end_s": round(end / fps, 2),
                "min_knee_angle": round(min_knee, 1),
                "max_forward_lean": round(float(np.nanmax(s["lean"][start : end + 1])), 1),
                "descent_s": round(descent_s, 2),
                "ascent_s": round(ascent_s, 2),
                "depth_pass": depth_pass,
                "badge_frame": bottom,
                "badge": ("DEPTH OK" if depth_pass else "HIGH", depth_pass),
            }
        )

    return {
        "series": s,
        "reps": kept_reps,
        "rep_metrics": rep_metrics,
        "fps": fps,
        "rep_count": len(rep_metrics),
        "lift": "squat",
        "primary_key": "knee",
    }


def _is_spurious_squat(end, bottom, ascent_s, descent_s, duration_s, knee_range) -> bool:
    """True for detector artifacts: micro-blips, shallow bends, or setup merged into a rep."""
    return (
        end <= bottom
        or duration_s < MIN_REP_S
        or knee_range < MIN_SQUAT_RANGE
        or ascent_s > MAX_ASCENT_S
        or descent_s > MAX_ASCENT_S
    )


# ---------------------------------------------------------------- deadlift

def _liftoff_frame(signal: np.ndarray, bottom: int, top: int) -> int:
    """The frame the bar breaks the floor = the rep START: the LAST frame still near the floor (the
    signal's min over the pull) before it rises to the lockout ``top``. The argmin valley alone can
    sit seconds earlier, on the floor during setup/rest. ``signal`` rises floor -> lockout (hip angle).
    """
    seg = signal[bottom : top + 1]
    if len(seg) == 0:
        return bottom
    floor, peak = float(np.nanmin(seg)), float(np.nanmax(seg))
    rom = peak - floor
    if rom <= 0:
        return bottom
    near = np.where(seg <= floor + 0.1 * rom)[0]
    return bottom + int(near[-1]) if len(near) else bottom


def _deadlift_lockout(top_lean: float, top_knee: float):
    """IPF 2026 deadlift lockout from the top-of-rep geometry: standing erect (torso ~vertical = the
    shoulder lined up over the hip) AND knees locked straight. Returns (hips_locked, knees_locked,
    lockout_pass). Pure — unit-tested."""
    hips_locked = bool(top_lean <= ERECT_LEAN_MAX)   # §4.3.1.2 stand erect / shoulders back
    knees_locked = bool(top_knee >= KNEE_LOCK_MIN)   # §4.3.1.3 knees straight
    return hips_locked, knees_locked, (hips_locked and knees_locked)


def analyze_deadlift(pose: P.PoseResult) -> dict:
    s = compute_series(pose)
    candidates = detect_reps(s["hip"], DL_STAND_HIP, DL_BOTTOM_HIP)
    fps = pose.fps
    n = pose.num_frames
    bottoms = [c["bottom"] for c in candidates]

    # Pass 1: TRUE lockout = peak hip angle from a rep's bottom up to the NEXT rep's bottom
    # (or end of clip). detect_reps closes a rep the instant hip first crosses the 'standing'
    # threshold — which is BEFORE full lockout — so we search past it for the real peak.
    kept = []
    for i, c in enumerate(candidates):
        start, bottom = c["start"], c["bottom"]
        win_end = bottoms[i + 1] if i + 1 < len(candidates) else n
        seg = s["hip"][bottom:win_end]
        if len(seg) == 0:
            continue
        top = bottom + int(np.nanargmax(seg))
        top_hip = float(s["hip"][top])
        amplitude = top_hip - float(s["hip"][bottom])
        ascent_s = (top - bottom) / fps
        duration_s = (top - start) / fps
        if _is_spurious_rep(top, bottom, ascent_s, duration_s, amplitude):
            continue
        kept.append({"start": start, "bottom": bottom, "top": top, "top_hip": top_hip,
                     "top_knee": float(s["knee"][top]), "top_lean": float(s["lean"][top])})

    # Pass 2: build metrics. Each rep effectively ends at its lockout frame (top). Lockout is judged
    # by standing erect (torso vertical = shoulder over the hip) + knees straight (see _deadlift_lockout).
    kept_reps = []
    rep_metrics = []
    for k in kept:
        start, bottom, top, top_hip = k["start"], k["bottom"], k["top"], k["top_hip"]
        top_knee, top_lean = k["top_knee"], k["top_lean"]
        liftoff = _liftoff_frame(s["hip"], bottom, top)   # bar breaks the floor = the rep start
        hips_locked, knees_locked, lockout_pass = _deadlift_lockout(top_lean, top_knee)
        hip_rise_ratio = _hip_rise_ratio(s, bottom, top)
        kept_reps.append({"start": start, "bottom": bottom, "liftoff": liftoff, "end": top})
        rep_metrics.append(
            {
                "start_s": round(start / fps, 2),
                "bottom_s": round(bottom / fps, 2),
                "top_s": round(top / fps, 2),
                "lockout_hip_angle": round(top_hip, 1),
                "lockout_knee_angle": round(top_knee, 1),
                "torso_lean_deg": round(top_lean, 1),
                "hips_locked": hips_locked,
                "knees_locked": knees_locked,
                "lockout_pass": lockout_pass,
                "hip_rise_ratio": None if hip_rise_ratio is None else round(hip_rise_ratio, 2),
                "descent_s": round((bottom - start) / fps, 2),
                "ascent_s": round((top - liftoff) / fps, 2),   # the pull only (excl. floor rest)
                "badge_frame": top,
                "badge": ("LOCKOUT" if lockout_pass else "INCOMPLETE", lockout_pass),
            }
        )

    return {
        "series": s,
        "reps": kept_reps,
        "rep_metrics": rep_metrics,
        "fps": fps,
        "rep_count": len(rep_metrics),
        "lift": "deadlift",
        "primary_key": "hip",
    }


def _is_spurious_rep(top, bottom, ascent_s, duration_s, amplitude) -> bool:
    """True for detector artifacts: micro-blips, trailing garbage, or setup merged into a rep."""
    return (
        top <= bottom
        or ascent_s <= 0
        or ascent_s > MAX_ASCENT_S
        or duration_s < MIN_REP_S
        or amplitude < MIN_REP_RANGE
    )


def _hip_rise_ratio(s: dict, bottom: int, top: int, frac: float = HIP_RISE_FRAC):
    """Vertical rise of the hip vs the shoulder during the early pull.

    ~1.0 = hips and shoulders rise together (good). >1 = hips shooting up first.
    Returns None when there isn't enough upward motion to judge.
    """
    if top <= bottom:
        return None
    early_end = bottom + max(1, int((top - bottom) * frac))
    d_hip = s["hip_y"][bottom] - s["hip_y"][early_end]          # up is positive (y down)
    d_shoulder = s["shoulder_y"][bottom] - s["shoulder_y"][early_end]
    if d_hip <= 0:
        return None
    if d_shoulder <= 1e-6:
        return float("inf")  # shoulders didn't rise at all while hips did
    return d_hip / d_shoulder
