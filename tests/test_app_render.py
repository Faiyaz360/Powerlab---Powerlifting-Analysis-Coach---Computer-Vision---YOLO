"""Score banner + leaderboard HTML rendering, including XSS-escaping of user-supplied names."""
import app
from src import history


def _sc(validated=True):
    return {"score": 88, "grade": "A", "validated": validated, "legal": True, "axis_ok": True,
            "best_rep": 1, "reason": "validated for the leaderboard",
            "breakdown": {"legality": 100, "technique": 80, "bar_path": 70, "control": 90,
                          "consistency": 60}}


def test_score_html_renders_value_grade_and_status():
    html = app._score_html(_sc())
    assert "88" in html and "/100" in html and "leaderboard" in html.lower()


def test_score_html_empty_when_no_score():
    assert app._score_html(None) == ""


def test_leaderboard_html_escapes_lifter_name():
    rows = [{"rank": 1, "lifter_name": "<script>alert(1)</script>", "lift": "squat",
             "bar_load_kg": 150.0, "score": 90.0, "grade": "A+", "dots": 120.0}]
    html = app._leaderboard_html(rows, "Score")
    assert "<script>" not in html                 # raw tag must not survive
    assert "&lt;script&gt;" in html               # it was escaped


def test_leaderboard_html_empty_state():
    assert "No validated lifts" in app._leaderboard_html([], "Score")


def test_load_board_end_to_end(tmp_path, monkeypatch):
    db = str(tmp_path / "h.db")
    monkeypatch.setattr(app, "DB_PATH", db)
    history.save_run(db, {"created_at": "2026-06-17T01:00:00", "lifter_name": "Ann", "lift": "squat",
                          "rep_count": 3, "bar_load_kg": 130.0, "score": 91.0, "grade": "A+",
                          "validated": 1, "sex": "female", "bodyweight_kg": 60.0, "dots": 110.0})
    html = app.load_board("Score", "")
    assert "Ann" in html and "91" in html


def test_db_snapshot_restore_survives_a_restart(tmp_path, monkeypatch):
    """Leaderboard DB snapshots to the mounted bucket and restores onto a fresh (post-restart) disk."""
    bucket = tmp_path / "bucket"
    bucket.mkdir()
    monkeypatch.setattr(app, "PERSIST_DIR", str(bucket))
    # boot 1: save a validated lift, snapshot to the bucket
    monkeypatch.setattr(app, "DB_PATH", str(tmp_path / "boot1" / "history.db"))
    history.save_run(app.DB_PATH, {"created_at": "2026-06-17T01:00:00", "lifter_name": "Ann",
                                   "lift": "squat", "rep_count": 3, "bar_load_kg": 130.0, "score": 91.0,
                                   "grade": "A+", "validated": 1, "sex": "male", "bodyweight_kg": 80.0})
    app._snapshot_db()
    assert (bucket / "history.db").exists()          # persisted to the bucket
    # boot 2: a brand-new empty local disk (Space restarted) -> restore from the bucket
    db2 = tmp_path / "boot2" / "history.db"
    monkeypatch.setattr(app, "DB_PATH", str(db2))
    app._restore_db()
    assert db2.exists()
    lb = history.leaderboard(str(db2), by="score")
    assert lb and lb[0]["lifter_name"] == "Ann"       # data survived the restart
