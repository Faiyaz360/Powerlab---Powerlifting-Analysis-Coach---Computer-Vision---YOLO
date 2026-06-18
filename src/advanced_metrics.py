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


# ---------------------------------------------------------------- strength tier (typed weights)
# DOTS coefficients are exact/published. The load-velocity %1RM model and the velocity->RPE
# tables below are GENERALIZED and CALIBRATABLE — approximate, caveated, tuned per lifter later.

_DOTS_COEF = {
    "male": (-0.000001093, 0.0007391293, -0.1918759221, 24.0900756, -307.75076),
    "female": (-0.0000010706, 0.0005158568, -0.1126655495, 13.6175032, -57.96288),
}


def dots(load_kg: float, bodyweight_kg: float, sex: str = "male") -> float | None:
    """DOTS strength score for a SINGLE lift (not a 3-lift total). Exact published coefficients."""
    if not load_kg or not bodyweight_kg:
        return None
    a, b, c, d, e = _DOTS_COEF.get(sex, _DOTS_COEF["male"])
    bw = bodyweight_kg
    denom = a * bw**4 + b * bw**3 + c * bw**2 + d * bw + e
    if denom == 0:
        return None
    return round(load_kg * 500.0 / denom, 1)


def est_1rm(load_kg, reps, last_mcv, lift) -> dict | None:
    """Estimate 1RM from the SET: the rep count plus how close the LAST rep was to failure (its bar
    speed -> RPE -> reps in reserve), via Brzycki on the projected reps-to-failure. Brzycki is
    accurate in powerlifting's low-rep range and, unlike Epley, returns the lifted load itself for a
    true single (Epley inflates a 1-rep max by 3.3%). Population RPE table — see _RPE_TABLE.

    e.g. 120 kg x 7 with a last rep at RPE 8 -> 2 in reserve -> 9 reps to failure -> ~154 kg.
    """
    if not load_kg or not reps or last_mcv is None or lift not in _RPE_TABLE:
        return None
    rpe = velocity_to_rpe(last_mcv, lift)
    if rpe is None:
        return None
    rir = max(0.0, 10.0 - rpe)                        # reps left in reserve on the last rep
    reps_to_failure = reps + rir
    e1rm = load_kg * 36.0 / (37.0 - reps_to_failure)  # Brzycki (returns the load itself at 1 rep)
    conf = "good" if rir <= 1 else ("medium" if rir <= 3 else "low")
    if lift == "deadlift" and conf == "good":
        conf = "medium"   # deadlift is the least reliable lift for velocity->1RM (Hooper 2017)
    return {"e1rm_kg": round(e1rm, 1), "confidence": conf, "rpe": round(rpe, 1)}


def peak_power_w(load_kg: float, peak_velocity: float | None) -> float | None:
    """Barbell peak power (W) ~= load * g * peak velocity. Ignores system mass/accel (approx)."""
    if not load_kg or peak_velocity is None:
        return None
    return round(load_kg * 9.81 * peak_velocity, 1)


# MCV->RPE per lift, RPE-10 anchored to the published true-1RM mean-concentric velocity (MVT):
# squat ~0.25-0.30, deadlift ~0.15-0.17 m/s (Helms 2017; Gonzalez-Badillo), ~0.04-0.05 m/s per RPE.
# Population DEFAULTS — per-lifter calibration via lv_profile is the accurate long-term anchor.
_RPE_TABLE = {
    "squat": ([0.25, 0.30, 0.35, 0.45, 0.55], [10, 9, 8, 7, 6]),
    "deadlift": ([0.15, 0.19, 0.23, 0.31, 0.39], [10, 9, 8, 7, 6]),
}


def velocity_to_rpe(mcv: float | None, lift: str) -> float | None:
    """Estimated RPE from mean concentric velocity (slower = higher RPE). Generalized."""
    if mcv is None or lift not in _RPE_TABLE:
        return None
    xs, rpe = _RPE_TABLE[lift]
    return round(float(np.interp(mcv, xs, rpe)), 1)


# General VBT training-zone bands by mean concentric velocity (m/s), descending. These label the
# speed-strength continuum (what quality the rep trained), distinct from RPE (how close to failure).
# Generalized across squat/deadlift — calibrate per lifter. (lower_bound, label)
_VELOCITY_ZONES = [(1.00, "Speed"), (0.75, "Speed-Strength"), (0.50, "Strength-Speed"),
                   (0.30, "Heavy Strength"), (0.00, "Max Strength")]


def velocity_zone(mcv: float | None) -> str | None:
    """Qualitative VBT zone from mean concentric velocity (m/s). None when there's no velocity."""
    if mcv is None:
        return None
    for lower, label in _VELOCITY_ZONES:
        if mcv >= lower:
            return label
    return "Max Strength"
