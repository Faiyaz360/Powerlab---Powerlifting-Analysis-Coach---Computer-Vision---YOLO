"""Strength TIER for a single lift — a gamified rank from the lift's DOTS, bracketed PER LIFT.

DOTS sees only load + bodyweight + sex, NOT which lift it was — but a given DOTS is harder to reach
on a squat than a deadlift, and far harder on a bench (people deadlift > squat > bench). So each lift
gets its own bands, calibrated to that lift's IPF (drug-tested) world records:
  - deadlift ceiling ~270 single-lift DOTS (Jessica Buettner 261.5 kg @ 67 kg ~= 271)
  - squat    ceiling ~250 (Austin Perkins 341 kg @ 74 kg ~= 247, Sara Naldi 197.5 kg @ 57 kg ~= 226)
  - bench    much lower (raw records sit well under squat/DL DOTS) — provisional until Phase 6.
A 3-lift TOTAL is a different beast (elite ~600-700; Kristy Hawkins 711) and gets its OWN scale, used
once a total board exists. HEURISTIC + CALIBRATABLE.
"""
from __future__ import annotations

TIERS = ["Beginner", "Intermediate", "Advanced", "Legendary", "Godly"]

# minimum DOTS to REACH each tier, per lift (Beginner..Godly)
_CUTS = {
    "deadlift": [0, 85, 135, 185, 230],
    "squat":    [0, 80, 125, 170, 215],
    "bench":    [0, 50, 85, 120, 160],    # provisional — recalibrate when bench (Phase 6) ships
    "total":    [0, 300, 400, 500, 600],  # 3-lift DOTS — SEPARATE scale, for a future total board
}
_FALLBACK = _CUTS["deadlift"]             # unknown lift -> deadlift bands


# IPF open weight-class upper limits (kg). Above the top limit = the "+" superheavy class.
_WEIGHT_CLASSES = {
    "male":   [59, 66, 74, 83, 93, 105, 120],
    "female": [47, 52, 57, 63, 69, 76, 84],
}


def weight_class(bodyweight_kg, sex) -> str | None:
    """IPF open weight class for a bodyweight + sex, e.g. '83 kg' or '120 kg+'. None if unknown.

    A lifter sits in the lightest class whose limit they don't exceed (74 kg -> '74 kg', 80 kg ->
    '83 kg'); above the heaviest limit is the superheavy '120 kg+' / '84 kg+'.
    """
    if not bodyweight_kg:
        return None
    limits = _WEIGHT_CLASSES.get((sex or "").lower())
    if not limits:
        return None
    for lim in limits:
        if bodyweight_kg <= lim:
            return f"{lim} kg"
    return f"{limits[-1]} kg+"


def tier(dots, lift=None) -> dict | None:
    """Tier from a single lift's DOTS, bracketed by THAT lift's bands, or None when DOTS is missing.

    ``lift`` = 'squat' / 'deadlift' / 'bench' / 'total' (unknown -> deadlift bands). Returns
    ``{tier, idx, dots, next, to_next, pct}`` — ``idx`` 0..4 (Beginner..Godly).
    """
    if dots is None:
        return None
    dots = float(dots)
    cuts = _CUTS.get((lift or "").lower(), _FALLBACK)
    idx = max(0, min(sum(1 for c in cuts if dots >= c) - 1, len(TIERS) - 1))
    nxt = TIERS[idx + 1] if idx < len(TIERS) - 1 else None
    if nxt is not None:
        lo, hi = cuts[idx], cuts[idx + 1]
        to_next = round(hi - dots, 1)
        pct = round(max(0.0, min(1.0, (dots - lo) / (hi - lo))), 2) if hi > lo else 0.0
    else:
        to_next, pct = 0.0, 1.0
    return {"tier": TIERS[idx], "idx": idx, "dots": round(dots, 1),
            "next": nxt, "to_next": to_next, "pct": pct}
