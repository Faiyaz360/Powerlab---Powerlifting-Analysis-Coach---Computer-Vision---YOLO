"""Load-velocity profile: line fit, inversion, 1RM estimate, and not-enough-data guards."""
import math

from src import lv_profile as lv


def test_fit_degenerate_velocities_is_not_a_perfect_fit():
    # distinct loads but identical velocities -> no real load-velocity relationship.
    # R2 must NOT read 1.0 (a degenerate flat fit was being reported as perfect).
    p = lv.fit_profile([(100, 0.5), (120, 0.5), (140, 0.5)])
    assert p is not None
    assert p["slope"] == 0.0
    assert p["r2"] == 0.0


def test_fit_perfect_line():
    # v drops 0.2 m/s per 20 kg -> slope -0.01, intercept 1.8
    p = lv.fit_profile([(100, 0.8), (120, 0.6), (140, 0.4)])
    assert p is not None
    assert math.isclose(p["slope"], -0.01, abs_tol=1e-9)
    assert math.isclose(p["intercept"], 1.8, abs_tol=1e-9)
    assert math.isclose(p["r2"], 1.0, abs_tol=1e-9)
    assert p["n"] == 3


def test_predictions_invert_correctly():
    p = lv.fit_profile([(100, 0.8), (140, 0.4)])
    assert math.isclose(lv.velocity_at_load(p, 120), 0.6, abs_tol=1e-9)
    assert math.isclose(lv.load_at_velocity(p, 0.5), 130.0, abs_tol=1e-9)
    # squat MVT 0.30 -> est 1RM 150 kg on this line
    assert math.isclose(lv.est_1rm(p, "squat"), 150.0, abs_tol=1e-9)


def test_needs_two_distinct_loads():
    assert lv.fit_profile([]) is None
    assert lv.fit_profile([(100, 0.8)]) is None
    assert lv.fit_profile([(100, 0.8), (100, 0.6)]) is None   # same load twice


def test_non_negative_slope_returns_no_load():
    # velocity rising with load (noise) -> can't invert
    p = lv.fit_profile([(100, 0.4), (140, 0.8)])
    assert lv.load_at_velocity(p, 0.5) is None
    assert lv.est_1rm(p, "squat") is None
