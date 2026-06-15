"""History repository tests against a temporary SQLite file."""
from src import history


def _record(**over):
    rec = {
        "created_at": "2026-06-15T10:00:00", "video_name": "squat1", "lift": "squat",
        "rep_count": 3, "depth_pass": 1, "mean_velocity": 0.42, "consistency": 92.0,
        "velocity_loss": 11.0, "annotated_path": "output/squat1_annotated.mp4",
    }
    rec.update(over)
    return rec


def test_save_then_get_roundtrip(tmp_path):
    db = str(tmp_path / "h.db")
    rid = history.save_run(db, _record())
    got = history.get_run(db, rid)
    assert got["video_name"] == "squat1"
    assert got["rep_count"] == 3
    # the forward-compat column defaults to 'side'
    assert got["view"] == "side"


def test_list_filters_by_lift(tmp_path):
    db = str(tmp_path / "h.db")
    history.save_run(db, _record(lift="squat"))
    history.save_run(db, _record(lift="deadlift", video_name="dl1"))
    squats = history.list_runs(db, lift="squat")
    assert len(squats) == 1
    assert squats[0]["lift"] == "squat"


def test_list_is_most_recent_first(tmp_path):
    db = str(tmp_path / "h.db")
    history.save_run(db, _record(video_name="first"))
    history.save_run(db, _record(video_name="second"))
    rows = history.list_runs(db)
    assert rows[0]["video_name"] == "second"


def test_trend_returns_time_value_pairs(tmp_path):
    db = str(tmp_path / "h.db")
    history.save_run(db, _record(created_at="2026-06-10T10:00:00", consistency=80.0))
    history.save_run(db, _record(created_at="2026-06-12T10:00:00", consistency=90.0))
    series = history.trend(db, "consistency", lift="squat")
    # oldest-first (id ascending) for a left-to-right time axis
    assert series == [("2026-06-10T10:00:00", 80.0), ("2026-06-12T10:00:00", 90.0)]


def test_trend_rejects_unknown_metric(tmp_path):
    db = str(tmp_path / "h.db")
    try:
        history.trend(db, "drop_table; --")
        assert False, "expected ValueError"
    except ValueError:
        pass
