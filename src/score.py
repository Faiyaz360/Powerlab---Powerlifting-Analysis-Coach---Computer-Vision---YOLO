"""Per-lift execution-quality score out of 100 + leaderboard-validity gate.

A robust, multi-factor score built only from what the pipeline already measures:

  legality (IPF depth / lockout) · technique faults · bar-path straightness · eccentric control ·
  rep-to-rep consistency

The score is the BEST rep's quality (the leaderboard stores one best lift per lifter). A lift is
``validated`` — eligible for the leaderboard — only when the camera is side-on (confident) AND the
best rep is IPF-legal; an off-axis or no-depth lift gets a score for feedback but never reaches the
board.

Pure (no I/O). Every component degrades gracefully: a missing component (e.g. no bar tracking) drops
out and the remaining weights renormalise, so the score stays out of 100 on whatever data exists.

Note: legality/technique are per-rep (pose `rep_metrics` + the fault list); bar-path/control/
consistency are set-level (shared by every rep). The best rep is the one with the best legality +
technique on top of the shared set-level quality.
"""
from __future__ import annotations

import numpy as np

from . import advanced_metrics as am
from . import faults as faultsmod

# Component weights (sum 1.0). Legality + technique are per-rep; the rest are set-level.
_W = {"legality": 0.35, "technique": 0.25, "bar_path": 0.15, "control": 0.15, "consistency": 0.10}
_DRIFT_BAD_CM = 15.0      # horizontal bar drift at which the bar-path sub-score hits 0
_COACHING_PENALTY = 0.4   # technique deduction per coaching fault on the rep
_CONTROL_FULL_RATIO = 0.6  # eccentric/concentric time at/above which control is full (not slammed down)
_GRADES = [(95, "S"), (90, "A+"), (80, "A"), (70, "B"), (60, "C"), (50, "D")]


def score_lift(analysis: dict, faults: dict | None = None, confidence: dict | None = None) -> dict | None:
    """Score the lift out of 100 (its best rep) and decide leaderboard validity.

    Returns ``None`` when there are no reps to score. Otherwise a dict:
    ``{score, grade, validated, legal, axis_ok, best_rep, breakdown, reason}``.
    """
    rep_metrics = analysis.get("rep_metrics") or []
    if not rep_metrics:
        return None
    lift = analysis.get("lift")
    coaching_by_rep = _coaching_by_rep(faults)
    # set-level sub-scores (same for every rep; None when their data is absent)
    bar_path = _bar_path_sub(analysis)
    control = _control_sub(analysis)
    consistency = _consistency_sub(analysis)

    best = None
    for i, rep in enumerate(rep_metrics):
        legality, legal = _legality_sub(rep, lift)
        technique = _technique_sub(coaching_by_rep.get(i, []))
        rep_score = _combine([
            (_W["legality"], legality),
            (_W["technique"], technique),
            (_W["bar_path"], bar_path),
            (_W["control"], control),
            (_W["consistency"], consistency),
        ])
        if rep_score is None:
            continue
        if best is None or rep_score > best["score"]:
            best = {
                "rep": i + 1, "score": rep_score, "legal": legal,
                "breakdown": {"legality": _pct(legality), "technique": _pct(technique),
                              "bar_path": _pct(bar_path), "control": _pct(control),
                              "consistency": _pct(consistency)},
            }
    if best is None:
        return None

    axis_ok = bool(confidence.get("axis_ok")) if confidence else False
    validated = bool(best["legal"] and axis_ok)
    return {
        "score": int(best["score"]),
        "grade": _grade(best["score"]),
        "validated": validated,
        "legal": best["legal"],
        "axis_ok": axis_ok,
        "best_rep": best["rep"],
        "breakdown": best["breakdown"],
        "reason": _reason(best["legal"], axis_ok, lift),
    }


# ---------------------------------------------------------------- component sub-scores (0..1)

def _legality_sub(rep: dict, lift: str | None):
    """IPF legality of the rep -> (value 0..1, legal bool). Squat = depth; deadlift = hips+knees."""
    if lift == "squat":
        ok = bool(rep.get("depth_pass"))
        return (1.0 if ok else 0.3), ok
    hips, knees = bool(rep.get("hips_locked")), bool(rep.get("knees_locked"))
    if hips and knees:
        return 1.0, True
    if hips or knees:
        return 0.6, False
    return 0.25, False


def _technique_sub(coaching_issues: list) -> float:
    """1.0 minus a deduction per coaching fault on the rep (forward lean, hips-rise, ...)."""
    return max(0.0, 1.0 - _COACHING_PENALTY * len(coaching_issues))


def _bar_path_sub(analysis: dict):
    """Straightness from the median per-rep horizontal drift; None when uncalibrated/untracked."""
    drifts = _rep_drifts(analysis)
    if not drifts:
        return None
    return float(np.clip(1.0 - float(np.median(drifts)) / _DRIFT_BAD_CM, 0.0, 1.0))


def _control_sub(analysis: dict):
    """Eccentric control: a dropped/slammed eccentric (tiny ecc vs con time) scores low. None when
    there are no calibrated reps with both phase times."""
    ratios = []
    for v in analysis.get("bar_velocity") or []:
        if not v:
            continue
        con, ecc = v.get("concentric_s"), v.get("eccentric_s")
        if con and ecc and con > 0:
            ratios.append(ecc / con)
    if not ratios:
        return None
    return float(np.clip(float(np.median(ratios)) / _CONTROL_FULL_RATIO, 0.0, 1.0))


def _consistency_sub(analysis: dict):
    """Rep-to-rep reproducibility (depth, ascent time, bar speed) as 0..1; None for a single rep."""
    rm = analysis.get("rep_metrics") or []
    depth_key = "min_knee_angle" if analysis.get("lift") == "squat" else "lockout_hip_angle"
    feats = {
        "depth": [r.get(depth_key) for r in rm],
        "ascent": [r.get("ascent_s") for r in rm],
        "mcv": [v.get("mean_velocity_ms") for v in (analysis.get("bar_velocity") or []) if v],
    }
    cs = am.consistency_score(feats)
    return None if cs is None else cs / 100.0


def _rep_drifts(analysis: dict) -> list:
    bar_xy, scale = analysis.get("bar_xy"), analysis.get("scale_m_per_px")
    if bar_xy is None or scale is None:
        return []
    out = []
    for r in analysis.get("bar_reps") or []:
        d = am.bar_path_drift(bar_xy, scale, r["bottom"], r["top"])
        if d:
            out.append(d["peak_drift_cm"])
    return out


def _coaching_by_rep(faults: dict | None) -> dict:
    """rep index (0-based) -> coaching-category issue keys on that rep (legal faults handled
    separately by the legality sub-score)."""
    if not faults:
        return {}
    out = {}
    for pr in faults.get("per_rep", []):
        out[pr["rep"] - 1] = [k for k in pr.get("issues", [])
                              if faultsmod.ISSUE_META.get(k, {}).get("category") == "coaching"]
    return out


# ---------------------------------------------------------------- combine + format

def _combine(parts: list):
    """Weighted mean of present (weight, value) parts, renormalised over what's present. None if
    nothing is present. Returns 0..100."""
    present = [(w, v) for w, v in parts if v is not None]
    total = sum(w for w, _ in present)
    if total <= 0:
        return None
    return round(100.0 * sum(w * v for w, v in present) / total, 0)


def _pct(v):
    return None if v is None else int(round(100 * v))


def _grade(score: float) -> str:
    for threshold, grade in _GRADES:
        if score >= threshold:
            return grade
    return "E"


def _reason(legal: bool, axis_ok: bool, lift: str | None) -> str:
    if not axis_ok:
        return "not on the leaderboard — camera off-axis (film dead side-on)"
    if not legal:
        return f"not on the leaderboard — best rep missed {'depth' if lift == 'squat' else 'lockout'}"
    return "validated for the leaderboard"
