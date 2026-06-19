"""Load-velocity (LV) profile: fit mean concentric velocity vs load, then predict load / velocity / 1RM.

VBT basis (Gonzalez-Badillo, Sanchez-Medina): across submaximal loads the mean concentric velocity
falls roughly LINEARLY as load rises. So fit ``v = slope*load + intercept`` from a lifter's logged
sets, then invert it to answer "what load moves at my target velocity today?" and estimate 1RM as the
load at the lift's minimal-velocity threshold (MVT ~ the mean velocity of a true 1RM).

Pure + deterministic — no I/O. The history DB supplies the (load, velocity) points.
"""
from __future__ import annotations

# Minimal-velocity threshold (m/s): the mean concentric velocity at a true 1RM, per lift.
MVT = {"squat": 0.30, "deadlift": 0.15}
DEFAULT_MVT = 0.30


def fit_profile(points: list[tuple]) -> dict | None:
    """Least-squares line ``v = slope*load + intercept`` from (load, velocity) points.

    Needs >= 2 points across >= 2 DISTINCT loads (a single load can't define a slope). Returns
    ``{slope, intercept, r2, n, load_min, load_max}`` or ``None`` when there isn't enough spread.
    """
    pts = [(float(l), float(v)) for l, v in points if l and v is not None]
    if len(pts) < 2 or len({round(l, 1) for l, _ in pts}) < 2:
        return None
    n = len(pts)
    xs = [l for l, _ in pts]
    ys = [v for _, v in pts]
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return None
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    # ss_tot == 0 means every velocity is identical -> no real load-velocity relationship; that's a
    # degenerate (flat) fit, NOT a perfect one, so report r2 = 0 rather than a misleading 1.0.
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"slope": slope, "intercept": intercept, "r2": r2, "n": n,
            "load_min": min(xs), "load_max": max(xs)}


def velocity_at_load(profile: dict, load: float) -> float:
    """Predicted mean concentric velocity (m/s) at a given load."""
    return profile["slope"] * load + profile["intercept"]


def load_at_velocity(profile: dict, velocity: float) -> float | None:
    """Invert the line: the load that should move at ``velocity``. ``None`` if the slope isn't negative
    (a flat or positive LV line is non-physical — too little data / noise)."""
    if not profile or profile["slope"] >= 0:
        return None
    return (velocity - profile["intercept"]) / profile["slope"]


def est_1rm(profile: dict, lift: str | None = None) -> float | None:
    """Estimated 1RM = the load at the lift's minimal-velocity threshold (extrapolated — treat as a
    guide, not a max attempt)."""
    return load_at_velocity(profile, MVT.get(lift or "", DEFAULT_MVT))


# --- per-lifter calibration: a lifter's OWN RPE-10 velocity, learned from their logged lifts -------
_MIN_CALIB_POINTS = 3      # logged lifts of this lift before we trust a personal MVT over the table
_HEAVY_FRAC = 0.85         # a rep counts as HEAVY if its load >= this * the lifter's heaviest logged
_MVT_BAND = (0.03, 0.60)   # plausible RPE-10 velocity (m/s); outside = mis-scaled video, reject


def personal_mvt(points: list[tuple], lift: str | None = None) -> float | None:
    """A lifter's OWN minimal-velocity threshold (their RPE-10 velocity), for per-lifter RPE/e1RM
    calibration. It is the SLOWEST mean concentric velocity they have produced on a HEAVY rep of this
    lift — you cannot move slower than your true grind, so that min approximates their personal MVT.

    ``points`` is their prior (load, velocity) history for the lift. Returns None until there is
    enough heavy data (cold start -> the caller falls back to the population table), or if the result
    is physically implausible (a mis-scaled plate). ``lift`` is accepted for a future per-lift band;
    the population default still lives in ``MVT``. Pure.
    """
    pts = [(float(l), float(v)) for l, v in points if l and v is not None and v > 0]
    if len(pts) < _MIN_CALIB_POINTS:
        return None
    max_load = max(l for l, _ in pts)
    heavy = [v for l, v in pts if l >= _HEAVY_FRAC * max_load]
    if not heavy:
        return None
    mvt = min(heavy)
    lo, hi = _MVT_BAND
    return mvt if lo <= mvt <= hi else None
