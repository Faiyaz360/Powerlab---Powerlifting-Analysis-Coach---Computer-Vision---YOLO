"""Concentric velocity loss is a NEUTRAL autoregulation metric, never a fault.

Set fatigue (reps slowing toward the end) is normal and expected — if reps DON'T slow, the load is
light or the lifter is sandbagging. So detect_faults reports `concentric_slowdown_pct` but never
adds a fault for it.
"""
from src.faults import detect_faults


def _dl_analysis(ascents, bar_velocity=None, hip_rise=1.0):
    reps = [{"hips_locked": True, "knees_locked": True, "hip_rise_ratio": hip_rise,
             "descent_s": 0.4, "ascent_s": a} for a in ascents]
    analysis = {"rep_count": len(reps), "rep_metrics": reps}
    if bar_velocity is not None:
        analysis["bar_velocity"] = bar_velocity
    return analysis


def test_velocity_loss_is_measured_but_never_a_fault():
    faults = detect_faults(_dl_analysis([0.8, 1.2, 1.71]), "deadlift")
    assert faults["concentric_slowdown_pct"] > 25                 # the number is reported
    assert "concentric_velocity_loss" not in faults["issue_list"]  # but it is NOT a fault
    assert "inconsistent_tempo" not in faults["issue_list"]        # nor the old id
    assert faults["issue_list"] == []                             # a clean steady-locked set: no faults


def test_slowdown_number_is_directional():
    sped_up = detect_faults(_dl_analysis([1.5, 1.1, 0.8]), "deadlift")
    assert sped_up["concentric_slowdown_pct"] <= 0                # got faster -> non-positive
    steady = detect_faults(_dl_analysis([1.0, 1.0, 1.0]), "deadlift")
    assert steady["concentric_slowdown_pct"] == 0


def test_slowdown_prefers_bar_velocity_over_pose_proxy():
    analysis = _dl_analysis([1.0, 1.0],
                            bar_velocity=[{"mean_velocity_ms": 0.5}, {"mean_velocity_ms": 0.3}])
    assert detect_faults(analysis, "deadlift")["concentric_slowdown_pct"] == 40.0


def test_real_faults_still_detected_alongside_the_metric():
    # a genuine fault (hips rise too fast) is still flagged; velocity loss stays a metric only
    faults = detect_faults(_dl_analysis([0.8, 1.71], hip_rise=12.25), "deadlift")
    assert "hips_rise_too_fast" in faults["issue_list"]
    assert "concentric_velocity_loss" not in faults["issue_list"]
