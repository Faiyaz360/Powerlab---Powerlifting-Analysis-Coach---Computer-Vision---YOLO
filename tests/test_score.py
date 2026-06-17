"""Execution-quality /100 score + leaderboard-validity gate (pure)."""
from src import score


def _clean_squat(axis_ok=True):
    analysis = {
        "lift": "squat",
        "rep_metrics": [
            {"depth_pass": True, "min_knee_angle": 70, "ascent_s": 1.0, "max_forward_lean": 30},
            {"depth_pass": True, "min_knee_angle": 72, "ascent_s": 1.1, "max_forward_lean": 31},
        ],
        "bar_xy": None, "scale_m_per_px": None, "bar_reps": [],
        "bar_velocity": [
            {"concentric_s": 1.0, "eccentric_s": 1.2, "mean_velocity_ms": 0.50},
            {"concentric_s": 1.1, "eccentric_s": 1.2, "mean_velocity_ms": 0.49},
        ],
    }
    faults = {"per_rep": [{"rep": 1, "issues": []}, {"rep": 2, "issues": []}]}
    return analysis, faults, {"axis_ok": axis_ok}


def test_clean_legal_sideon_squat_scores_high_and_validates():
    s = score.score_lift(*_clean_squat(axis_ok=True))
    assert s["legal"] is True
    assert s["validated"] is True
    assert s["score"] >= 85 and s["grade"] in ("S", "A+", "A")


def test_missed_depth_is_not_validated_and_not_legal():
    analysis, faults, conf = _clean_squat()
    for r in analysis["rep_metrics"]:
        r["depth_pass"] = False
    s = score.score_lift(analysis, faults, conf)
    assert s["legal"] is False
    assert s["validated"] is False
    assert s["score"] < 85                       # legality dragged it down


def test_offaxis_legal_lift_scores_but_is_not_validated():
    s = score.score_lift(*_clean_squat(axis_ok=False))
    assert s["legal"] is True                     # the rep was legal...
    assert s["axis_ok"] is False
    assert s["validated"] is False                # ...but we can't trust an off-axis camera


def test_coaching_fault_lowers_score_vs_clean():
    clean = score.score_lift(*_clean_squat())
    analysis, faults, conf = _clean_squat()
    faults["per_rep"][0]["issues"] = ["excessive_forward_lean"]
    faults["per_rep"][1]["issues"] = ["excessive_forward_lean"]
    flawed = score.score_lift(analysis, faults, conf)
    assert flawed["score"] < clean["score"]


def test_score_degrades_gracefully_without_bar_data():
    # single clean rep, no bar tracking: only legality + technique present -> still a valid /100
    analysis = {"lift": "squat", "bar_xy": None, "scale_m_per_px": None, "bar_reps": [],
                "bar_velocity": [], "rep_metrics": [{"depth_pass": True, "ascent_s": 1.0}]}
    s = score.score_lift(analysis, {"per_rep": [{"rep": 1, "issues": []}]}, {"axis_ok": True})
    assert s is not None and s["score"] == 100 and s["validated"] is True


def test_deadlift_locked_out_is_legal_and_validates():
    analysis = {"lift": "deadlift", "bar_xy": None, "scale_m_per_px": None, "bar_reps": [],
                "bar_velocity": [], "rep_metrics": [
                    {"hips_locked": True, "knees_locked": True, "lockout_hip_angle": 172, "ascent_s": 1.4}]}
    s = score.score_lift(analysis, {"per_rep": [{"rep": 1, "issues": []}]}, {"axis_ok": True})
    assert s["legal"] is True and s["validated"] is True


def test_no_reps_returns_none():
    assert score.score_lift({"lift": "squat", "rep_metrics": []}, None, None) is None
