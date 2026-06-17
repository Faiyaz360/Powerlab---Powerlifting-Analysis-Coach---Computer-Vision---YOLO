"""Unit tests for derived lift metrics (pure functions, deterministic)."""
import numpy as np

from src import advanced_metrics as am


def test_velocity_loss_basic():
    # Arrange: first rep 0.50 m/s, last rep 0.40 m/s
    bar_velocity = [{"mean_velocity_ms": 0.50}, {"mean_velocity_ms": 0.45}, {"mean_velocity_ms": 0.40}]
    # Act
    loss = am.velocity_loss_pct(bar_velocity)
    # Assert: (0.50 - 0.40) / 0.50 * 100 = 20.0
    assert loss == 20.0


def test_velocity_zone_bands():
    assert am.velocity_zone(1.2) == "Speed"
    assert am.velocity_zone(0.85) == "Speed-Strength"
    assert am.velocity_zone(0.60) == "Strength-Speed"
    assert am.velocity_zone(0.40) == "Heavy Strength"
    assert am.velocity_zone(0.20) == "Max Strength"
    assert am.velocity_zone(None) is None


def test_velocity_loss_needs_two_reps():
    assert am.velocity_loss_pct([{"mean_velocity_ms": 0.5}]) is None


def test_velocity_loss_ignores_none_reps():
    bar_velocity = [{"mean_velocity_ms": 0.5}, None, {"mean_velocity_ms": 0.4}]
    assert am.velocity_loss_pct(bar_velocity) == 20.0


def test_bar_path_drift_cm_and_direction():
    # Arrange: x drifts +3 px from start; scale 0.005 m/px -> 3 * 0.005 * 100 = 1.5 cm
    bar_xy = np.array([[100.0, 0.0], [101.0, 0.0], [103.0, 0.0], [101.0, 0.0]])
    # Act
    drift = am.bar_path_drift(bar_xy, 0.005, 0, 3)
    # Assert
    assert drift == {"peak_drift_cm": 1.5, "direction": "forward"}


def test_bar_path_drift_none_without_scale():
    bar_xy = np.array([[100.0, 0.0], [103.0, 0.0]])
    assert am.bar_path_drift(bar_xy, None, 0, 1) is None


def test_consistency_high_when_reps_alike():
    # Arrange: depth varies a hair, ROM identical -> very consistent
    feature_series = {"depth": [90.0, 92.0, 88.0], "rom": [0.50, 0.50, 0.50]}
    # Act
    score = am.consistency_score(feature_series)
    # Assert: mean CV ~= 0.00907 -> 100*(1-0.00907) ~= 99
    assert score == 99.0


def test_consistency_none_with_one_rep():
    assert am.consistency_score({"depth": [90.0]}) is None


def test_sticking_point_finds_slow_patch():
    # Arrange: plate rises with a slow patch around index 4 (height units; y = -height).
    height = np.array([0, 2, 4, 6, 6.2, 8, 10, 12, 14], dtype=float)
    bar_y = -height
    # Act
    sp = am.sticking_point_pct(bar_y, 0, 8)
    # Assert: slowest ascent at index 4 -> 6.2/14*100 = 44; frame index 4
    assert sp == {"pct_of_rom": 44.0, "frame_idx": 4}


def test_sticking_point_none_when_no_rise():
    bar_y = np.zeros(9)
    assert am.sticking_point_pct(bar_y, 0, 8) is None


# --- strength tier (C-tier) ---

def test_dots_male_reference():
    # 200 kg lift at 100 kg bodyweight, male coefficients
    assert am.dots(200.0, 100.0, "male") == 123.1


def test_dots_none_without_weights():
    assert am.dots(0, 100.0) is None


def test_est_1rm_uses_reps_and_last_rep_rpe():
    # 120 kg x 7, last rep 0.30 m/s -> RPE 8 -> 2 in reserve -> 9 reps to failure -> Epley.
    out = am.est_1rm(120.0, 7, 0.30, "squat")
    assert round(out["e1rm_kg"]) == 156   # 120 * (1 + 9/30)
    assert out["rpe"] == 8.0
    assert out["confidence"] == "medium"


def test_est_1rm_good_confidence_near_failure():
    # last rep 0.20 m/s -> RPE 10 -> 0 in reserve -> high confidence.
    assert am.est_1rm(140.0, 5, 0.20, "squat")["confidence"] == "good"


def test_est_1rm_low_confidence_fast_last_rep():
    # last rep 0.50 m/s -> RPE 6 -> 4 in reserve -> low confidence (extrapolating far).
    assert am.est_1rm(100.0, 3, 0.50, "squat")["confidence"] == "low"


def test_est_1rm_none_without_velocity_or_reps():
    assert am.est_1rm(140.0, 5, None, "squat") is None
    assert am.est_1rm(140.0, 0, 0.3, "squat") is None


def test_peak_power():
    assert am.peak_power_w(100.0, 1.0) == 981.0


def test_velocity_to_rpe_squat():
    assert am.velocity_to_rpe(0.30, "squat") == 8.0
    assert am.velocity_to_rpe(0.45, "squat") == 6.5
