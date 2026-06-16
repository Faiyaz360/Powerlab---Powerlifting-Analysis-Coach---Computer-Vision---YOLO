"""1-euro path smoothing used by the optical-flow bar tracker (src/barbell.py)."""
import numpy as np

from src.barbell import _one_euro


def test_preserves_length_and_first_sample():
    # Arrange
    x = np.array([10.0, 12.0, 9.0, 11.0, 10.5])
    # Act
    y = _one_euro(x)
    # Assert: same number of frames, anchored to the first sample (no warm-up offset).
    assert len(y) == len(x)
    assert y[0] == x[0]


def test_constant_signal_unchanged():
    # Arrange: a perfectly still plate.
    x = np.full(30, 100.0)
    # Act
    y = _one_euro(x)
    # Assert: nothing to smooth -> stays put.
    assert np.allclose(y, 100.0)


def test_reduces_jitter_around_a_level():
    # Arrange: a steady centre with frame-to-frame noise (the jitter we want gone).
    base = 200.0
    noise = np.array([+3, -4, +2, -3, +4, -2, +3, -4, +2, -3, +4, -2], dtype=float)
    x = base + noise
    # Act
    y = _one_euro(x)
    # Assert: smoothed path varies less than the noisy input.
    assert np.var(y) < np.var(x)


def test_tracks_a_ramp_without_runaway():
    # Arrange: a steadily rising signal (bar moving up through the pull).
    x = np.arange(40, dtype=float) * 2.0
    # Act
    y = _one_euro(x)
    # Assert: output stays within the data's range (no overshoot/instability) and keeps rising.
    assert y.min() >= x.min() - 1e-6
    assert y.max() <= x.max() + 1e-6
    assert y[-1] > y[0]
