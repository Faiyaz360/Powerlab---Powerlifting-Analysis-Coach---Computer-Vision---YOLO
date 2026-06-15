"""Derived lift metrics computed from the analysis dict the pipeline already produces.

Every function is pure (no I/O) and returns ``None`` when there isn't enough data, so the UI
can show a "needs more reps" placeholder instead of crashing. Heuristics — calibrate per lifter.
"""
from __future__ import annotations

import numpy as np


def velocity_loss_pct(bar_velocity: list) -> float | None:
    """Drop from the first to the last rep's mean concentric velocity, as a percent.

    VBT fatigue proxy: ~20-25% loss signals proximity to failure. Needs >=2 calibrated reps.
    """
    mcvs = [v["mean_velocity_ms"] for v in bar_velocity
            if v and v.get("mean_velocity_ms") is not None]
    if len(mcvs) < 2 or mcvs[0] <= 0:
        return None
    return round((mcvs[0] - mcvs[-1]) / mcvs[0] * 100, 1)


def bar_path_drift(bar_xy: np.ndarray, scale_m_per_px: float | None,
                   start: int, end: int) -> dict | None:
    """Peak horizontal deviation of the plate from its start-x over a rep, in cm.

    ``direction`` is image-relative ("forward" = +x) in v1; mapping to the lifter's facing is a
    later refinement. Returns None when uncalibrated (no scale) or too few points.
    """
    if scale_m_per_px is None:
        return None
    xs = bar_xy[start:end + 1, 0].astype(float)
    xs = xs[np.isfinite(xs)]
    if len(xs) < 2:
        return None
    dev = xs - xs[0]
    i = int(np.argmax(np.abs(dev)))
    return {
        "peak_drift_cm": round(abs(dev[i]) * scale_m_per_px * 100, 1),
        "direction": "forward" if dev[i] > 0 else "back",
    }


def consistency_score(feature_series: dict) -> float | None:
    """0-100 reproducibility score from the coefficient of variation across reps.

    ``feature_series`` maps a feature name (depth angle, ROM, tempo, ...) to its per-rep values.
    Lower variation -> higher score. Needs >=2 reps for at least one feature. Heuristic.
    """
    cvs = []
    for values in feature_series.values():
        vals = [v for v in values if v is not None and np.isfinite(v)]
        if len(vals) < 2:
            continue
        mean = float(np.mean(vals))
        if mean == 0:
            continue
        cvs.append(float(np.std(vals)) / abs(mean))
    if not cvs:
        return None
    score = 100.0 * (1.0 - float(np.mean(cvs)))
    return round(max(0.0, min(100.0, score)), 0)


def sticking_point_pct(bar_y: np.ndarray, bottom: int, top: int) -> dict | None:
    """Where in the ascent the bar is slowest, as a percent of ROM from the bottom.

    Searches the middle 20-80% of the concentric (ignoring start/end noise) for the minimum
    upward velocity. ``bar_y`` is the plate's vertical pixel position (down is +). Returns the
    %-of-ROM and the absolute frame index for a video marker, or None if there's no clean ascent.
    """
    seg = bar_y[bottom:top + 1].astype(float)
    if len(seg) < 5:
        return None
    height = -seg  # up is positive
    rom = height[-1] - height[0]
    if rom <= 0:
        return None
    vel = np.gradient(height)
    lo, hi = int(0.2 * len(seg)), int(0.8 * len(seg))
    if hi <= lo:
        return None
    i_local = lo + int(np.argmin(vel[lo:hi]))
    pct = (height[i_local] - height[0]) / rom * 100
    return {"pct_of_rom": round(float(pct), 0), "frame_idx": bottom + i_local}
