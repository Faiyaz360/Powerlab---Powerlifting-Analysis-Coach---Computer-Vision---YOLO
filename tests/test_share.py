"""Share-clip tests: caption building (pure) + portrait clip dimensions (round-trips a tiny video)."""
import cv2
import numpy as np

from src import share


def test_caption_full_lift_has_handle_numbers_and_tags():
    meta = {"lift": "squat", "load": 140, "score": 88, "grade": "A+",
            "legal_pass": True, "peak_ms": 0.42, "reps": 3}
    cap = share.share_caption(meta)
    assert "@projectfyz" in cap
    assert "Squat @ 140kg" in cap
    assert "Score 88/100 (A+)" in cap
    assert "Depth: passed" in cap
    assert "0.42 m/s" in cap
    assert "#powerlifting" in cap and "#squat" in cap


def test_caption_deadlift_says_lockout():
    cap = share.share_caption({"lift": "deadlift", "legal_pass": False})
    assert "Lockout: missed" in cap
    assert "#deadlift" in cap


def test_caption_minimal_never_crashes():
    cap = share.share_caption({"lift": "squat"})
    assert "@projectfyz" in cap            # always tagged
    assert "Squat" in cap
    assert "kg" not in cap.split("\n")[0]  # no load -> no "@ kg" in the headline


def _write_tiny_video(path, w=320, h=240, frames=6, fps=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(frames):
        frame = np.full((h, w, 3), (i * 30) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_make_share_clip_outputs_portrait_video(tmp_path):
    src = tmp_path / "lift.mp4"
    _write_tiny_video(src)

    out = share.make_share_clip(src, tmp_path, lift="squat", score=88, grade="A+")

    cap = cv2.VideoCapture(out)
    assert cap.isOpened()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, frame = cap.read()
    cap.release()
    assert (width, height) == (share.OUT_W, share.OUT_H)   # 9:16 portrait
    assert ok and frame is not None


def test_make_share_clip_bad_source_raises(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        share.make_share_clip(tmp_path / "nope.mp4", tmp_path, lift="squat")
