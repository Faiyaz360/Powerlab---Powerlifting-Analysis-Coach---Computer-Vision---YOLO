"""Honesty layer: flag when the camera is too off-axis (or landmarks too weak) to trust the
depth/lockout verdict.

Side-on is required for valid 2D squat/deadlift geometry — off-axis foreshortens every angle
(see LESSONS.md). Rather than give a confident wrong depth call, the dashboard shows
"can't tell — camera off-axis" when this layer is not confident.
"""
from __future__ import annotations

import numpy as np

from . import pose as P

SIDEON_RATIO_MAX = 0.35   # below this = plausibly side-on (CALIBRATE on the eval clips)
MIN_VISIBILITY = 0.5
_KEY_LANDMARKS = [P.L_SHOULDER, P.R_SHOULDER, P.L_HIP, P.R_HIP, P.L_KNEE, P.R_KNEE]


def offaxis_ratio(landmarks: np.ndarray) -> float:
    """Mean L/R horizontal separation of shoulders+hips, divided by torso height.

    Side-on -> ~0 (the near and far landmarks overlap in x). Front/angled -> large. The ratio is
    scale-invariant, so it works for normalized or pixel coordinates.
    """
    ratios = []
    for f in range(len(landmarks)):
        lsh = landmarks[f, P.L_SHOULDER, :2]
        rsh = landmarks[f, P.R_SHOULDER, :2]
        lhip = landmarks[f, P.L_HIP, :2]
        rhip = landmarks[f, P.R_HIP, :2]
        if any(np.any(np.isnan(p)) for p in (lsh, rsh, lhip, rhip)):
            continue
        sh_sep = abs(lsh[0] - rsh[0])
        hip_sep = abs(lhip[0] - rhip[0])
        torso_h = abs((lsh[1] + rsh[1]) / 2 - (lhip[1] + rhip[1]) / 2)
        if torso_h < 1e-6:
            continue
        ratios.append(((sh_sep + hip_sep) / 2) / torso_h)
    return float(np.mean(ratios)) if ratios else float("nan")


def _mean_visibility(landmarks: np.ndarray) -> float:
    col = landmarks[:, _KEY_LANDMARKS, 2]
    if not np.any(np.isfinite(col)):
        return 0.0
    return float(np.nanmean(col))


def assess(pose, analysis=None) -> dict:
    """Confidence in the depth/lockout verdict, from camera angle + landmark visibility.

    Returns {level: 'high'|'low', reason, axis_ok, offaxis_ratio}. ``analysis`` is accepted for
    forward compatibility (future per-rep checks) but not yet used.
    """
    lm = pose.landmarks
    ratio = offaxis_ratio(lm)
    vis = _mean_visibility(lm)
    axis_ok = bool(np.isfinite(ratio) and ratio < SIDEON_RATIO_MAX)
    if vis < MIN_VISIBILITY:
        level, reason = "low", "landmarks poorly visible"
    elif not axis_ok:
        level, reason = "low", "camera looks off-axis — depth/lockout unreliable"
    else:
        level, reason = "high", "camera looks side-on"
    return {
        "level": level,
        "reason": reason,
        "axis_ok": axis_ok,
        "offaxis_ratio": None if not np.isfinite(ratio) else round(ratio, 2),
    }
