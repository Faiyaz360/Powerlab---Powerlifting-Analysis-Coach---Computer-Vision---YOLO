"""Strength TIER for a single lift — a gamified rank derived from the lift's DOTS score.

DOTS already normalizes load for bodyweight + sex (see advanced_metrics.dots), so the tier is just
a DOTS-band bracket: Beginner -> Godly. One scale for everyone — a woman's and a man's lift at the
same DOTS land in the same tier. HEURISTIC + CALIBRATABLE single-lift DOTS cuts (real DOTS is on the
3-lift total, so these are tuned for one lift and adjustable later). A deadlift naturally scores a
touch higher than a squat at equal effort (heavier absolute load) — fine for a motivating badge.
"""
from __future__ import annotations

TIERS = ["Beginner", "Intermediate", "Advanced", "Legendary", "Godly"]
# minimum (single-lift) DOTS to REACH each tier. ANCHORED TO REAL RECORDS: the highest single-lift
# DOTS ever are ~300 — Kristy Hawkins 310 kg squat @ 75 kg ~= 302, Andrzej Stanaszek 300.5 kg squat
# @ 50 kg ~= 299; elite heavier lifters ~260-270 (John Haack 410 kg deadlift @ 90 kg ~= 266). So
# Godly (200+) is genuine world-class and a strong amateur tops out Legendary. Tiers describe where
# you sit in the TRAINABLE POPULATION (a 1.9x-bodyweight deadlift is Advanced even though pros are at
# 260+). On a DEADLIFT the cuts land near 1x / 1.5x / 2.5x / 3.5x bodyweight (squat needs a touch
# more, women ~0.8x). HEURISTIC + CALIBRATABLE.
_CUTS = [0, 50, 90, 140, 200]


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
