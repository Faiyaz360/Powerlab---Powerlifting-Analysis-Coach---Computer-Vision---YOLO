"""3D Stage 1 — the COCO/Blaze -> H36M-17 adapter (pure; synthesises the spine chain)."""
import numpy as np

from src import lift3d
from src import pose as P


def _frame():
    xy = np.full((33, 3), np.nan)
    xy[P.L_HIP] = [100, 200, 1]; xy[P.R_HIP] = [120, 200, 1]
    xy[P.L_SHOULDER] = [100, 100, 1]; xy[P.R_SHOULDER] = [120, 100, 1]
    xy[P.NOSE] = [110, 60, 1]
    xy[P.L_KNEE] = [100, 280, 1]; xy[P.R_KNEE] = [120, 280, 1]
    xy[P.L_ANKLE] = [100, 360, 1]; xy[P.R_ANKLE] = [120, 360, 1]
    return xy


def test_to_h36m17_shape_and_order():
    out = lift3d.to_h36m17(np.stack([_frame()]))
    assert out.shape == (1, 17, 2)
    assert np.allclose(out[0, 0], [110, 200])    # Hip/pelvis = mid-hips
    assert np.allclose(out[0, 8], [110, 100])    # Thorax = mid-shoulders
    assert np.allclose(out[0, 7], [110, 150])    # Spine = mid(pelvis, thorax)
    assert np.allclose(out[0, 1], [120, 200])    # RHip 1:1
    assert np.allclose(out[0, 4], [100, 200])    # LHip 1:1
    assert np.allclose(out[0, 11], [100, 100])   # LShoulder 1:1
    assert np.allclose(out[0, 14], [120, 100])   # RShoulder 1:1


def test_to_h36m17_head_extrapolated_above_nose():
    out = lift3d.to_h36m17(np.stack([_frame()]))
    # head = nose + (nose - thorax) * 0.3 = (110,60) + ((110,60)-(110,100))*0.3 = (110, 48)
    assert np.allclose(out[0, 9], [110, 60])     # Neck ~ nose
    assert np.allclose(out[0, 10], [110, 48])    # Head a little above the nose


def test_to_h36m17_missing_joint_propagates_nan():
    f = _frame()
    f[P.R_WRIST] = [np.nan, np.nan, np.nan]      # never tracked
    out = lift3d.to_h36m17(np.stack([f]))
    assert np.all(np.isnan(out[0, 16]))          # RWrist stays NaN
    assert not np.any(np.isnan(out[0, 0]))       # pelvis still fine


def test_fill_missing_interpolates_and_flags():
    h = np.full((3, 17, 2), np.nan)
    h[0, 1] = [0, 0]; h[2, 1] = [10, 20]         # joint 1 seen at t0 and t2 -> t1 interpolates
    filled, present = lift3d.fill_missing(h)
    assert np.allclose(filled[1, 1], [5, 10])    # midpoint
    assert present[0, 1] and present[2, 1] and not present[1, 1]
    assert not present[:, 5].any()               # joint 5 never seen ...
    assert np.allclose(filled[:, 5], 0)          # ... and zero-filled, no NaN left


def test_normalize_crop_maps_bbox_to_unit_range():
    m = np.zeros((1, 17, 3))
    m[0, :4] = [[0, 0, 1], [10, 0, 1], [0, 10, 1], [10, 10, 1]]   # a 10x10 square, conf 1
    out = lift3d.normalize_crop(m)
    assert np.allclose(out[0, 0, :2], [-1, -1])  # bbox min -> -1
    assert np.allclose(out[0, 3, :2], [1, 1])    # bbox max -> +1
    assert out.min() >= -1 and out.max() <= 1
