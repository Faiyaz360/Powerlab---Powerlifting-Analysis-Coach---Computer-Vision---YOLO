"""Back-axis tracker: draws the hip->shoulder line + angle, and no-ops when landmarks are missing."""
import numpy as np

from src import pose as P
from src import render


def _one_frame_landmarks():
    lm = np.full((1, 33, 3), np.nan, dtype=float)
    lm[0, P.L_SHOULDER] = [150, 120, 1.0]   # upper back
    lm[0, P.L_HIP] = [150, 240, 1.0]        # low back, directly below
    return lm


def test_back_line_paints_spine_colour():
    frame = np.zeros((400, 300, 3), dtype=np.uint8)
    render._draw_back_line(frame, _one_frame_landmarks(), 0, "left", None, 8.0)
    b, g, r = render.SPINE
    painted = np.any((frame[:, :, 0] == b) & (frame[:, :, 1] == g) & (frame[:, :, 2] == r))
    assert painted, "back-axis line/markers should paint SPINE-coloured pixels"


def test_back_line_noop_when_landmarks_missing():
    frame = np.zeros((400, 300, 3), dtype=np.uint8)
    lm = np.full((1, 33, 3), np.nan, dtype=float)   # nothing detected
    render._draw_back_line(frame, lm, 0, "left", None, None)
    assert not frame.any(), "nothing to draw -> frame untouched"
