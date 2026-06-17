"""Confidence layer: side-on vs off-axis synthetic landmark sets."""
import numpy as np

from src import confidence as conf
from src import pose as P


def _frame(lsh_x, rsh_x, lhip_x, rhip_x, vis=1.0, n_landmarks=33):
    """One-frame landmark array with shoulders at y=0.2, hips at y=0.5, knees at y=0.8."""
    lm = np.full((1, n_landmarks, 3), np.nan)
    lm[0, P.L_SHOULDER] = [lsh_x, 0.2, vis]
    lm[0, P.R_SHOULDER] = [rsh_x, 0.2, vis]
    lm[0, P.L_HIP] = [lhip_x, 0.5, vis]
    lm[0, P.R_HIP] = [rhip_x, 0.5, vis]
    lm[0, P.L_KNEE] = [lhip_x, 0.8, vis]
    lm[0, P.R_KNEE] = [rhip_x, 0.8, vis]
    return lm


class _Pose:
    def __init__(self, landmarks):
        self.landmarks = landmarks


def test_offaxis_ratio_small_when_side_on():
    lm = _frame(0.5, 0.5, 0.5, 0.5)  # L/R overlap in x
    assert conf.offaxis_ratio(lm) < 0.05


def test_offaxis_ratio_large_when_front_on():
    lm = _frame(0.4, 0.6, 0.4, 0.6)  # wide L/R separation
    assert conf.offaxis_ratio(lm) > 0.5


def test_offaxis_ratio_robust_when_bent_over_side_on():
    """A dead-side-on but BENT-OVER pose (deadlift): L/R overlap in x, torso near-horizontal. The old
    vertical-height denominator exploded the ratio (false off-axis); torso-length keeps it low."""
    lm = np.full((1, 33, 3), np.nan)
    lm[0, P.L_SHOULDER] = [0.30, 0.50, 1.0]      # shoulders forward, hips back, ~same height
    lm[0, P.R_SHOULDER] = [0.31, 0.50, 1.0]      # L/R overlap in x = side-on
    lm[0, P.L_HIP] = [0.60, 0.51, 1.0]
    lm[0, P.R_HIP] = [0.61, 0.51, 1.0]
    assert conf.offaxis_ratio(lm) < conf.SIDEON_RATIO_MAX   # side-on, not false-flagged


def test_assess_high_confidence_side_on():
    out = conf.assess(_Pose(_frame(0.5, 0.5, 0.5, 0.5)))
    assert out["axis_ok"] is True
    assert out["level"] == "high"


def test_assess_low_confidence_off_axis():
    out = conf.assess(_Pose(_frame(0.4, 0.6, 0.4, 0.6)))
    assert out["axis_ok"] is False
    assert out["level"] == "low"


def test_assess_low_when_poor_visibility():
    out = conf.assess(_Pose(_frame(0.5, 0.5, 0.5, 0.5, vis=0.1)))
    assert out["level"] == "low"
