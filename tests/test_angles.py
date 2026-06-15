"""Unit tests for the angle math (AAA: Arrange-Act-Assert)."""
import math

from src.angles import angle_from_vertical, calc_angle


def test_right_angle_returns_90():
    # Arrange
    a = (0, 1)
    b = (0, 0)
    c = (1, 0)
    # Act
    angle = calc_angle(a, b, c)
    # Assert
    assert math.isclose(angle, 90.0, abs_tol=1e-6)


def test_straight_line_returns_180():
    assert math.isclose(calc_angle((0, 0), (1, 0), (2, 0)), 180.0, abs_tol=1e-6)


def test_folded_back_returns_0():
    # both rays point the same direction from the vertex
    assert math.isclose(calc_angle((1, 0), (0, 0), (1, 0)), 0.0, abs_tol=1e-6)


def test_45_degree_angle():
    assert math.isclose(calc_angle((1, 0), (0, 0), (1, 1)), 45.0, abs_tol=1e-6)


def test_degenerate_returns_nan():
    # vertex coincides with a point -> zero-length segment
    assert math.isnan(calc_angle((0, 0), (0, 0), (1, 0)))


def test_vertical_segment_is_zero_lean():
    # straight up (remember: image y grows downward, so 'top' has the smaller y)
    assert math.isclose(angle_from_vertical((0, 0), (0, 1)), 0.0, abs_tol=1e-6)


def test_horizontal_segment_is_90_lean():
    assert math.isclose(angle_from_vertical((1, 0), (0, 0)), 90.0, abs_tol=1e-6)
