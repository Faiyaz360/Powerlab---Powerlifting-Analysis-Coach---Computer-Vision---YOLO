"""Ghost compare — the pure pack/align math (real landmarks; honest)."""
import json

import numpy as np

from src import ghost
from src import pose as P


def _frame():
    xy = np.full((33, 3), np.nan)
    xy[P.L_SHOULDER] = [100, 100, 1]; xy[P.R_SHOULDER] = [120, 100, 1]
    xy[P.L_HIP] = [100, 200, 1]; xy[P.R_HIP] = [120, 200, 1]
    return xy


def test_build_blob_packs_key_frame():
    lm = np.stack([_frame(), _frame()])
    blob = json.loads(ghost.build_blob(lm, 1, "left", np.array([[110, 150], [110, 160]])))
    assert blob["side"] == "left"
    assert blob["hip"] == [110.0, 200.0]      # midpoint of the two hips
    assert blob["torso"] == 100.0             # shoulder y=100 -> hip y=200
    assert len(blob["bar"]) == 2


def test_build_blob_none_on_bad_frame():
    lm = np.stack([_frame()])
    assert ghost.build_blob(lm, 5, "left", None) is None          # key frame out of range
    assert ghost.build_blob(np.full((1, 33, 3), np.nan), 0, "left", None) is None  # no pose


def test_align_maps_hip_and_scales():
    blob = {"xy": [[110, 200], [110, 100]], "hip": [110, 200], "torso": 100, "bar": []}
    xy, bar = ghost.align(blob, [300, 300], 200)                  # current hip + torso (scale x2)
    assert np.allclose(xy[0], [300, 300])                          # ghost hip -> current hip
    assert np.allclose(xy[1], [300, 100])                          # 100 above hip, scaled x2 -> 200 above
    assert bar.shape == (0, 2)


def test_align_keeps_missing_joints_nan():
    blob = {"xy": [[110, 200], [None, None]], "hip": [110, 200], "torso": 100, "bar": []}
    xy, _ = ghost.align(blob, [0, 0], 100)
    assert np.allclose(xy[0], [0, 0]) and np.all(np.isnan(xy[1]))


def _leaned_frame():
    """Same hips, but shoulders shifted forward — a different SHAPE, so it won't overlap the current."""
    xy = _frame()
    xy[P.L_SHOULDER] = [140, 100, 1]; xy[P.R_SHOULDER] = [160, 100, 1]
    return xy


def test_draw_ghost_panel_paints_both_reps():
    frame = np.zeros((400, 400, 3), np.uint8)
    blob = ghost.build_blob(np.stack([_leaned_frame()]), 0, "left", None)   # best-ever = leaned
    img = ghost.draw_ghost_panel(frame, _frame(), "left", blob)             # current = upright
    assert img is not None
    cb, cg, cr = ghost._CUR_COLOR
    gb, gg, gr = ghost._GHOST_COLOR
    assert np.any((img[:, :, 0] == cb) & (img[:, :, 1] == cg) & (img[:, :, 2] == cr))   # current
    assert np.any((img[:, :, 0] == gb) & (img[:, :, 1] == gg) & (img[:, :, 2] == gr))   # ghost


def test_draw_ghost_panel_none_without_current_pose():
    frame = np.zeros((400, 400, 3), np.uint8)
    assert ghost.draw_ghost_panel(frame, np.full((33, 3), np.nan), "left",
                                  '{"xy":[],"hip":[0,0],"torso":1,"side":"left","bar":[]}') is None


def test_history_stores_and_fetches_best_ghost(tmp_path):
    from src import history
    db = str(tmp_path / "h.db")
    base = {"created_at": "2026-06-18T00:00:00", "lifter_name": "Ann", "lift": "squat",
            "validated": 1, "score": 80, "ghost": '{"v":1}'}
    history.save_run(db, base)
    history.save_run(db, {**base, "created_at": "2026-06-18T01:00:00", "score": 92, "ghost": '{"v":2}'})
    assert history.best_ghost(db, "Ann", "squat") == '{"v":2}'   # the higher-scoring run's ghost
    assert history.best_ghost(db, "Ann", "deadlift") is None     # none of that lift
    assert history.best_ghost(db, "Bo", "squat") is None         # unknown lifter


def test_align_accepts_json_string():
    s = ghost.build_blob(np.stack([_frame()]), 0, "left", None)
    xy, _ = ghost.align(s, [0, 0], 100)                            # hip [110,200] -> [0,0]
    assert np.allclose(xy[P.L_HIP], [-10, 0])                      # L_HIP [100,200] is 10 left of hip
