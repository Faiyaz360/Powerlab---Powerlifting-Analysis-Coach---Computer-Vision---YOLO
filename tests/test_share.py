"""Share caption tests (pure text built from the lift's numbers; always tags @projectfyz)."""
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
    assert "PowerLab" in cap


def test_caption_deadlift_says_lockout():
    cap = share.share_caption({"lift": "deadlift", "legal_pass": False})
    assert "Lockout: missed" in cap
    assert "#deadlift" in cap


def test_caption_minimal_never_crashes():
    cap = share.share_caption({"lift": "squat"})
    assert "@projectfyz" in cap                  # always tagged
    assert "Squat" in cap
    assert "kg" not in cap.split("\n")[0]         # no load -> no "@ kg" in the headline
