"""Deterministic fault detection: metrics -> a structured issue list.

This is the ONLY place faults are decided. The coaching layer (coach.py) just phrases these;
it never invents new ones. Each fault is tagged as a COMPETITION-LEGAL issue (would draw a red
light) or a COACHING issue (technique, not legality), with the governing rule.

Rule basis: IPF Technical Rulebook (effective 01 Mar 2026). Thresholds are measurement-calibrated
approximations of the written rules — tune per lifter/camera.
"""
from __future__ import annotations

import numpy as np

# squat
FORWARD_LEAN_MAX = 55.0   # degrees from vertical at any point in the rep (coaching)
# deadlift
HIP_RISE_RATIO_MAX = 1.4  # hips rising >1.4x faster than shoulders early in the pull (coaching)
# shared
TEMPO_CV_MAX = 0.25       # coefficient of variation of rep durations across the set (coaching)

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
    "inconsistent_tempo": {
        "category": "coaching", "lift": "both",
        "rule": "Coaching — consistent rep tempo reflects control",
    },
}


def detect_faults(analysis: dict, lift: str) -> dict:
    if lift == "squat":
        issues, per_rep = _squat_faults(analysis)
    elif lift == "deadlift":
        issues, per_rep = _deadlift_faults(analysis)
    else:
        raise NotImplementedError(f"No fault rules for lift '{lift}' yet.")

    tempo_cv = _tempo_cv(analysis["rep_metrics"])
    if tempo_cv is not None and tempo_cv > TEMPO_CV_MAX:
        issues.append("inconsistent_tempo")

    issue_list = sorted(set(issues))
    return {
        "issue_list": issue_list,
        "legal_issues": [i for i in issue_list if ISSUE_META.get(i, {}).get("category") == "legal"],
        "coaching_issues": [i for i in issue_list if ISSUE_META.get(i, {}).get("category") == "coaching"],
        "per_rep": per_rep,
        "tempo_cv": None if tempo_cv is None else round(tempo_cv, 3),
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


def _tempo_cv(rep_metrics):
    durations = [r["descent_s"] + r["ascent_s"] for r in rep_metrics]
    if len(durations) < 2:
        return None
    mean = float(np.mean(durations))
    return float(np.std(durations) / mean) if mean > 0 else 0.0
