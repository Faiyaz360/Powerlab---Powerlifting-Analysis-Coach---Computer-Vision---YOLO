"""Stage 2a draw: the back silhouette curve paints SPINE-coloured pixels along the given points."""
import numpy as np

from src import render


def test_back_curve_paints_spine_colour_along_the_points():
    frame = np.zeros((400, 300, 3), dtype=np.uint8)
    curve = [(120, 100), (122, 150), (124, 200), (126, 250)]   # a gentle back curve
    render._draw_back_curve(frame, curve, lean_deg=9.0)
    b, g, r = render.SPINE
    painted = np.any((frame[:, :, 0] == b) & (frame[:, :, 1] == g) & (frame[:, :, 2] == r))
    assert painted, "back-curve polyline/markers should paint SPINE-coloured pixels"


def test_back_curve_colours_a_bend_warm():
    """A sharp bend paints a warm (reddish) segment; a straight stretch stays violet."""
    frame = np.zeros((400, 300, 3), dtype=np.uint8)
    curve = [(100, 100), (100, 150), (100, 200), (150, 200), (200, 200)]   # ~90° corner at index 2
    render._draw_back_curve(frame, curve, None)
    warm = np.any((frame[:, :, 2] > 200) & (frame[:, :, 0] < 80))          # high R + low B = hot
    assert warm, "the bend should colour a warm/red segment"


def test_spine_curve_returns_none_without_extractor():
    """No extractor (feature off / model missing) -> None, so the caller falls back to Stage 1."""
    lm = np.zeros((1, 33, 3), dtype=float)
    out = render._spine_curve(None, np.zeros((10, 10, 3), np.uint8), lm, 0, "left", None, None, None)
    assert out is None
