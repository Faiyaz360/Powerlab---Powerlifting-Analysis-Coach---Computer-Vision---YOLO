"""Unit tests for rep detection (pure function, deterministic — no video needed)."""
import numpy as np

from src.metrics import detect_reps

STAND = 160.0
BOTTOM = 140.0


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
