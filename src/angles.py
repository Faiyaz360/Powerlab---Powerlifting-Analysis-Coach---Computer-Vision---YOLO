"""Pure geometry helpers — joint angles from 2D points. Unit-tested in tests/test_angles.py.

All points are (x, y) in PIXEL coordinates (image origin top-left, y increasing downward).
We use pixels, not MediaPipe's normalized 0..1 coords, because normalized x and y are scaled
differently (by width vs height) which would distort every angle.
"""
from __future__ import annotations

import numpy as np


def calc_angle(a, b, c) -> float:
    """Angle at vertex ``b`` formed by points a-b-c, in degrees (0..180).

    Returns NaN if any segment has zero length (degenerate / missing point).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return float("nan")
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def angle_from_vertical(top, bottom) -> float:
    """How far the segment ``bottom -> top`` tilts from vertical, in degrees.

    0 = perfectly vertical (e.g. an upright torso), 90 = horizontal. Used for forward-lean.
    """
    top = np.asarray(top, dtype=float)
    bottom = np.asarray(bottom, dtype=float)
    v = top - bottom
    horizontal = abs(v[0])
    vertical = abs(v[1])
    if horizontal == 0 and vertical == 0:
        return float("nan")
    return float(np.degrees(np.arctan2(horizontal, vertical)))
