"""Strength TIER for a single lift — a gamified rank derived from the lift's DOTS score.

DOTS already normalizes load for bodyweight + sex (see advanced_metrics.dots), so the tier is just
a DOTS-band bracket: Beginner -> Godly. One scale for everyone — a woman's and a man's lift at the
same DOTS land in the same tier. HEURISTIC + CALIBRATABLE single-lift DOTS cuts (real DOTS is on the
3-lift total, so these are tuned for one lift and adjustable later). A deadlift naturally scores a
touch higher than a squat at equal effort (heavier absolute load) — fine for a motivating badge.
"""
from __future__ import annotations

TIERS = ["Beginner", "Intermediate", "Advanced", "Legendary", "Godly"]
_CUTS = [0, 50, 80, 110, 140]          # minimum (single-lift) DOTS to REACH each tier


def tier(dots) -> dict | None:
    """Tier from a DOTS score, or None when DOTS is missing.

    Returns ``{tier, idx, dots, next, to_next, pct}`` — ``idx`` 0..4 (Beginner..Godly), ``to_next``
    is DOTS points to the next tier, ``pct`` is progress (0..1) through the current band.
    """
    if dots is None:
        return None
    dots = float(dots)
    idx = max(0, min(sum(1 for c in _CUTS if dots >= c) - 1, len(TIERS) - 1))
    nxt = TIERS[idx + 1] if idx < len(TIERS) - 1 else None
    if nxt is not None:
        lo, hi = _CUTS[idx], _CUTS[idx + 1]
        to_next = round(hi - dots, 1)
        pct = round(max(0.0, min(1.0, (dots - lo) / (hi - lo))), 2) if hi > lo else 0.0
    else:
        to_next, pct = 0.0, 1.0
    return {"tier": TIERS[idx], "idx": idx, "dots": round(dots, 1),
            "next": nxt, "to_next": to_next, "pct": pct}
