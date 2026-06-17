"""Deterministic fault detection: metrics -> a structured issue list.

This is the ONLY place faults are decided. The coaching layer (coach.py) just phrases these;
it never invents new ones. Each fault is tagged as a COMPETITION-LEGAL issue (would draw a red
light) or a COACHING issue (technique, not legality), with the governing rule.

Rule basis: IPF Technical Rulebook (effective 01 Mar 2026). Thresholds are measurement-calibrated
approximations of the written rules — tune per lifter/camera.
"""
from __future__ import annotations

import numpy as np

from . import advanced_metrics as am

# squat
FORWARD_LEAN_MAX = 55.0   # degrees from vertical at any point in the rep (coaching)
# deadlift
HIP_RISE_RATIO_MAX = 1.4  # hips rising >1.4x faster than shoulders early in the pull (coaching)
# NOTE: concentric velocity loss across a set is NOT a fault — it's normal fatigue (an
# autoregulation dial, goal-dependent). We REPORT it (concentric_slowdown_pct) but never flag it.

# key -> {category, lift, rule}. category: "legal" (red-light in competition) or "coaching".
ISSUE_META = {
    "depth_fail": {
        "category": "legal", "lift": "squat",
        "rule": "IPF 2026 §4.1.1.5 — hip crease must drop below the top of the knee",
    },
    "excessive_forward_lean": {
        "category": "coaching", "lift": "squat",
        "rule": "Coaching — keep the bar over midfoot; excessive trunk lean shifts load forward",
    },
    "hips_not_locked": {
        "category": "legal", "lift": "deadlift",
        "rule": "IPF 2026 §4.3.1.2 — must stand erect with the shoulders back",
    },
    "knees_not_locked": {
        "category": "legal", "lift": "deadlift",
        "rule": "IPF 2026 §4.3.1.3 — knees must be locked straight at completion",
    },
    "hips_rise_too_fast": {
        "category": "coaching", "lift": "deadlift",
        "rule": "Coaching (Starting Strength / JTS) — hips and shoulders should rise together",
    },
}


def detect_faults(analysis: dict, lift: str) -> dict:
    if lift == "squat":
        issues, per_rep = _squat_faults(analysis)
    elif lift == "deadlift":
        issues, per_rep = _deadlift_faults(analysis)
    else:
        raise NotImplementedError(f"No fault rules for lift '{lift}' yet.")

    issue_list = sorted(set(issues))
    return {
        "issue_list": issue_list,
        "legal_issues": [i for i in issue_list if ISSUE_META.get(i, {}).get("category") == "legal"],
        "coaching_issues": [i for i in issue_list if ISSUE_META.get(i, {}).get("category") == "coaching"],
        "per_rep": per_rep,
        # neutral autoregulation metric (reported, never a fault — see note at top of file)
        "concentric_slowdown_pct": _concentric_slowdown_pct(analysis),
        "rep_count": analysis["rep_count"],
    }


def _squat_faults(analysis):
    issues, per_rep = [], []
    for i, r in enumerate(analysis["rep_metrics"], 1):
        rep_issues = []
        if not r["depth_pass"]:
            rep_issues.append("depth_fail")
        if r["max_forward_lean"] > FORWARD_LEAN_MAX:
            rep_issues.append("excessive_forward_lean")
        per_rep.append({"rep": i, "issues": rep_issues})
        issues.extend(rep_issues)
    return issues, per_rep


def _deadlift_faults(analysis):
    issues, per_rep = [], []
    for i, r in enumerate(analysis["rep_metrics"], 1):
        rep_issues = []
        if not r["hips_locked"]:
            rep_issues.append("hips_not_locked")
        if not r["knees_locked"]:
            rep_issues.append("knees_not_locked")
        ratio = r["hip_rise_ratio"]
        if ratio is not None and ratio > HIP_RISE_RATIO_MAX:
            rep_issues.append("hips_rise_too_fast")
        per_rep.append({"rep": i, "issues": rep_issues})
        issues.extend(rep_issues)
    return issues, per_rep


def _concentric_slowdown_pct(analysis):
    """Directional concentric slowdown first->last rep, as a percent — a NEUTRAL autoregulation
    metric (set fatigue is normal, never a fault). Prefers calibrated bar velocity (mean concentric
    m/s); falls back to pose ascent duration (same ROM, so a longer ascent = a slower rep).
    Positive = slowed across the set; <=0 = held or sped up; None when there are <2 usable reps.
    """
    bar_loss = am.velocity_loss_pct(analysis.get("bar_velocity") or [])
    if bar_loss is not None:
        return bar_loss
    ascents = [r["ascent_s"] for r in analysis["rep_metrics"]
               if r.get("ascent_s") is not None and r["ascent_s"] > 0]
    if len(ascents) < 2:
        return None
    first, last = ascents[0], ascents[-1]
    return round((last - first) / last * 100, 1)  # velocity-loss proxy: 1 - v_last/v_first
