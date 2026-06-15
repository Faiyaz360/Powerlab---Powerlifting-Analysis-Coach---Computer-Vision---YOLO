"""Charts return matplotlib Figures without error from minimal synthetic data."""
import numpy as np
from matplotlib.figure import Figure

from src import charts


def _fake_analysis():
    return {
        "primary_key": "knee",
        "fps": 30.0,
        "series": {"knee": np.array([170, 120, 90, 120, 170], dtype=float)},
        "reps": [{"start": 0, "bottom": 2, "end": 4}],
    }


def test_angle_curve_returns_figure():
    assert isinstance(charts.angle_curve(_fake_analysis()), Figure)


def test_velocity_bars_returns_figure():
    bar_velocity = [{"mean_velocity_ms": 0.5}, {"mean_velocity_ms": 0.4}]
    assert isinstance(charts.velocity_bars(bar_velocity), Figure)


def test_velocity_bars_handles_empty():
    assert isinstance(charts.velocity_bars([]), Figure)


def test_drift_curve_handles_uncalibrated():
    assert isinstance(charts.drift_curve(None, None, 0, 1), Figure)
