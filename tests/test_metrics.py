"""Unit tests for rep detection (pure function, deterministic — no video needed)."""
import numpy as np

from src.metrics import _deadlift_lockout, _liftoff_frame, detect_reps

STAND = 160.0
BOTTOM = 140.0


def test_deadlift_lockout_needs_erect_torso_and_locked_knees():
    # erect (small torso lean = shoulder over hip) + knees straight -> locked out
    assert _deadlift_lockout(top_lean=5.0, top_knee=168.0) == (True, True, True)
    # still leaning forward at the top (hips not through) -> not erect -> no lockout
    assert _deadlift_lockout(top_lean=25.0, top_knee=168.0)[2] is False
    # erect but the knee is clearly bent (soft lockout) -> no lockout
    assert _deadlift_lockout(top_lean=5.0, top_knee=140.0)[2] is False


def test_liftoff_skips_floor_rest_to_the_break_off():
    # hip sits near the floor (60) through a long rest, then rises to lockout (170): liftoff is the
    # LAST near-floor frame (the bar breaking the ground), not the argmin valley at the start.
    sig = np.array([60, 60, 60, 60, 60, 60, 60, 80, 120, 170], dtype=float)
    assert _liftoff_frame(sig, 0, 9) == 6        # index 6 = last frame still at the floor before the rise


def test_single_rep_found_with_correct_bottom():
    # Arrange: stand -> dip to 90 -> stand
    signal = np.array([170, 170, 165, 120, 90, 120, 165, 170, 170], dtype=float)
    # Act
    reps = detect_reps(signal, STAND, BOTTOM)
    # Assert
    assert len(reps) == 1
    assert reps[0]["bottom"] == 4  # index of the minimum (90)


def test_two_reps():
    signal = np.array([170, 120, 170, 100, 170], dtype=float)
    reps = detect_reps(signal, STAND, BOTTOM)
    assert len(reps) == 2


def test_shallow_dip_not_counted():
    # dips to 150 — below 'stand' but never past 'bottom' (140), so not a rep
    signal = np.array([170, 170, 150, 170, 170], dtype=float)
    assert detect_reps(signal, STAND, BOTTOM) == []


def test_trailing_rep_when_video_ends_at_bottom():
    signal = np.array([170, 120, 90], dtype=float)
    reps = detect_reps(signal, STAND, BOTTOM)
    assert len(reps) == 1
    assert reps[0]["bottom"] == 2
