"""Strength-standard TIER for a single lift — where a load sits for the lifter's bodyweight + sex.

Bodyweight-ratio thresholds (Beginner -> Elite), per lift and sex. This is the friendly,
recognizable "is this strong?" layer — lifters think in ×BW — while the DOTS leaderboard is the
rigorous cross-bodyweight RANKING. HEURISTIC + CALIBRATABLE: ExRx / StrengthLevel-style ratios,
approximate, and (like the velocity->RPE tables) tunable per lifter later. Flat ratios slightly
favour heavier lifters at the same tier, so for the fair pound-for-pound comparison use DOTS; this
tier answers the softer "how am I doing for my weight class?".
"""
from __future__ import annotations

TIERS = ["Beginner", "Novice", "Intermediate", "Advanced", "Elite"]

# minimum (load / bodyweight) ratio to REACH each tier, by (sex, lift)
_RATIOS = {
    ("male", "squat"):      [1.00, 1.25, 1.50, 2.00, 2.50],
    ("male", "deadlift"):   [1.25, 1.50, 1.75, 2.25, 2.75],
    ("female", "squat"):    [0.70, 0.90, 1.10, 1.50, 1.90],
    ("female", "deadlift"): [0.90, 1.10, 1.35, 1.75, 2.10],
}


def tier(load_kg, bodyweight_kg, sex, lift) -> dict | None:
    """Strength tier for a single lift, or None when sex/lift is unknown or inputs are missing.

    Returns ``{tier, idx, ratio, next, to_next_kg, pct}`` — ``idx`` 0..4 (Beginner..Elite),
    ``pct`` is progress (0..1) toward the next tier (1.0 at Elite).
    """
    key = (str(sex).lower(), str(lift).lower())
    if key not in _RATIOS or not load_kg or not bodyweight_kg:
        return None
    r = _RATIOS[key]
    ratio = load_kg / bodyweight_kg
    idx = max(0, min(sum(1 for t in r if ratio >= t) - 1, len(TIERS) - 1))
    nxt = TIERS[idx + 1] if idx < len(TIERS) - 1 else None
    if nxt is not None:
        lo, hi = r[idx] * bodyweight_kg, r[idx + 1] * bodyweight_kg
        to_next_kg = round(hi - load_kg, 1)
        pct = round(max(0.0, min(1.0, (load_kg - lo) / (hi - lo))), 2) if hi > lo else 0.0
    else:
        to_next_kg, pct = 0.0, 1.0
    return {"tier": TIERS[idx], "idx": idx, "ratio": round(ratio, 2),
            "next": nxt, "to_next_kg": to_next_kg, "pct": pct}
