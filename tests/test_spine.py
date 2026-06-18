"""Spine Stage 2a (silhouette back-curve) — the PURE geometry, tested on synthetic masks so no
model or video is needed. The segmentation itself (MediaPipe selfie model) is exercised only for
its graceful-unavailable path."""
import numpy as np

from src import spine


def _torso_mask(h=200, w=200):
    """A simple rectangular 'person': columns 80..119, rows 20..179 are body."""
    m = np.zeros((h, w), bool)
    m[20:180, 80:120] = True
    return m


def test_back_contour_tracks_the_edge_opposite_the_knee():
    """Back side = opposite the knee. Knee on the LEFT -> the back contour rides the RIGHT edge."""
    m = _torso_mask()
    sh, hip = (100, 30), (100, 170)        # torso axis = vertical centre line
    knee_left = (60, 175)
    pts = spine.back_contour(m, sh, hip, knee_left)
    assert pts is not None and len(pts) >= 3
    xs = [p[0] for p in pts]
    assert min(xs) > 110                    # every station sits on the RIGHT (back) edge ~x=119


def test_back_contour_flips_with_the_knee():
    """Knee on the RIGHT -> the back is now the LEFT edge (proves the direction is knee-driven)."""
    m = _torso_mask()
    pts = spine.back_contour(m, (100, 30), (100, 170), (140, 175))
    assert pts is not None
    xs = [p[0] for p in pts]
    assert max(xs) < 90                     # every station on the LEFT (back) edge ~x=80


def test_back_contour_needs_real_axis():
    """Degenerate (shoulder == hip) -> no axis -> None, never a crash."""
    m = _torso_mask()
    assert spine.back_contour(m, (100, 100), (100, 100), (60, 120)) is None


def test_subtract_disc_removes_the_plate_and_is_immutable():
    """The barbell plate fuses into the person mask; subtracting its disc clears that circle and
    leaves a NEW array (the original mask is untouched)."""
    m = np.ones((50, 50), bool)
    out = spine.subtract_disc(m, (25, 25, 10))
    assert out[25, 25] == False             # plate centre cleared
    assert out[0, 0] == True                # far corner kept
    assert m[25, 25] == True                # original unchanged (immutable-style)


def test_subtract_disc_noop_without_radius():
    m = np.ones((10, 10), bool)
    assert np.array_equal(spine.subtract_disc(m, (5, 5, 0)), m)


def test_smooth_reduces_jaggedness_and_keeps_length():
    """Light moving-average along the contour cuts the staircase wobble of a raw silhouette edge."""
    zig = [(0, 0), (10, 0), (0, 0), (10, 0), (0, 0)]
    out = spine.smooth(zig)
    assert len(out) == len(zig)
    tv_raw = sum(abs(zig[i][0] - zig[i - 1][0]) for i in range(1, len(zig)))
    tv_out = sum(abs(out[i][0] - out[i - 1][0]) for i in range(1, len(out)))
    assert tv_out < tv_raw                   # smoother than the input


def test_curvature_zero_on_a_straight_contour():
    pts = [(0, i * 10) for i in range(6)]              # a straight vertical line
    c = spine.curvature(pts)
    assert len(c) == 6
    assert all(abs(x) < 1e-6 for x in c)              # nothing bends -> all 0


def test_curvature_peaks_at_a_bend():
    pts = [(0, 0), (0, 10), (0, 20), (10, 20), (20, 20)]   # L-shape: ~90° corner at index 2
    c = spine.curvature(pts)
    assert c.index(max(c)) == 2 and max(c) > 80       # the corner is the sharpest point


def test_curvature_short_input_is_safe():
    assert spine.curvature([(1, 1), (2, 2)]) == [0.0, 0.0]
    assert spine.curvature(None) == []


def test_backcurve_unavailable_is_graceful():
    """No model file -> available() is False and curve() returns None (caller falls back to the
    straight Stage-1 line) — the pipeline never crashes on a segmentation problem."""
    bc = spine.BackCurve(model_path="does/not/exist.tflite")
    assert bc.available() is False
    dummy = np.zeros((40, 40, 3), np.uint8)
    assert bc.curve(dummy, (10, 5), (10, 35), (5, 38)) is None
