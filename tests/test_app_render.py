"""Score banner + leaderboard HTML rendering, including XSS-escaping of user-supplied names."""
import app
from src import history
from src import strength_standards as ss


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


def test_leaderboard_html_dots_board_headlines_dots():
    """The DOTS board shows DOTS as the primary number + bodyweight in the subtitle (not a
    duplicate DOTS suffix)."""
    rows = [{"rank": 1, "lifter_name": "Lil", "lift": "deadlift", "bar_load_kg": 200.0,
             "score": 90.0, "grade": "A", "dots": 110.0, "bodyweight_kg": 70}]
    html = app._leaderboard_html(rows, "DOTS")
    assert "110" in html and "DOTS" in html        # DOTS is the headline
    assert "70 kg BW" in html                       # bodyweight shown for context
    assert html.count("DOTS") == 1                  # not repeated as a subtitle suffix


def test_leaderboard_dots_board_shows_strength_tier_chip():
    """On the DOTS board the gamified (DOTS-based) tier replaces the execution grade chip."""
    rows = [{"rank": 1, "lifter_name": "Lil", "lift": "deadlift", "bar_load_kg": 140.0,
             "score": 90.0, "grade": "A", "dots": 95.0, "bodyweight_kg": 70, "sex": "male"}]
    html = app._leaderboard_html(rows, "DOTS")
    assert "lb-tier" in html and "Advanced" in html        # DOTS 95 -> Advanced band
    assert "lb-grade" not in html                          # tier replaces the grade on this board


def test_leaderboard_row_shows_reps_and_e1rm():
    """A multi-rep set must not read like a 1RM: show '142 kg × 3' + the estimated single."""
    rows = [{"rank": 1, "lifter_name": "Faiyaz", "lift": "deadlift", "bar_load_kg": 142.0,
             "rep_count": 3, "e1rm_kg": 155.0, "score": 94.0, "grade": "S", "dots": 98.0,
             "bodyweight_kg": 80, "sex": "male"}]
    html = app._leaderboard_html(rows, "DOTS")
    assert "142 kg × 3" in html and "e1RM 155" in html


def test_leaderboard_godly_chip_is_animated():
    """A top-tier DOTS gets the animated shimmer chip + sparkle."""
    rows = [{"rank": 1, "lifter_name": "Hercules", "lift": "deadlift", "bar_load_kg": 320.0,
             "score": 96.0, "grade": "S", "dots": 160.0, "bodyweight_kg": 80, "sex": "male"}]
    html = app._leaderboard_html(rows, "DOTS")
    assert "lb-tier-godly" in html and "Godly" in html and "✨" in html


def test_leaderboard_legendary_chip_has_golden_glow():
    """Legendary gets the pulsing golden-glow chip (distinct from Godly's shimmer)."""
    rows = [{"rank": 1, "lifter_name": "Atlas", "lift": "deadlift", "bar_load_kg": 240.0,
             "score": 94.0, "grade": "S", "dots": 125.0, "bodyweight_kg": 80, "sex": "male"}]
    html = app._leaderboard_html(rows, "DOTS")
    assert "lb-tier-legendary" in html and "Legendary" in html
    assert "lb-tier-godly" not in html                     # not the top tier


def test_tier_card_legendary_glows():
    html = app._tier_card(ss.tier(125))                    # DOTS 125 -> Legendary
    assert "fl-tier-legendary" in html and "Legendary" in html


def test_tier_card_shows_dots_and_progress():
    html = app._tier_card(ss.tier(95))                     # Advanced, 15 DOTS to Legendary
    assert "Advanced" in html and "95 DOTS" in html and "15 to Legendary" in html
    assert app._TIER_HEX[2] in html                        # Advanced colour


def test_tier_card_godly_is_animated():
    html = app._tier_card(ss.tier(160))
    assert "fl-tier-godly" in html and "Godly" in html and "✨" in html


def test_tier_card_empty_when_no_tier():
    assert "—" in app._tier_card(None)


def test_load_board_end_to_end(tmp_path, monkeypatch):
    db = str(tmp_path / "h.db")
    monkeypatch.setattr(app, "DB_PATH", db)
    history.save_run(db, {"created_at": "2026-06-17T01:00:00", "lifter_name": "Ann", "lift": "squat",
                          "rep_count": 3, "bar_load_kg": 130.0, "score": 91.0, "grade": "A+",
                          "validated": 1, "sex": "female", "bodyweight_kg": 60.0, "dots": 110.0})
    html = app.load_board("Score", "")
    assert "Ann" in html and "91" in html


def test_session_csv_has_summary_and_per_rep_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "OUT_DIR", str(tmp_path))
    a = {"lift": "squat", "rep_count": 2, "bar_velocity": [
        {"concentric_s": 1.0, "mean_velocity_ms": 0.50, "peak_velocity_ms": 0.80,
         "eccentric_s": 1.2, "rom_m": 0.52},
        {"concentric_s": 1.1, "mean_velocity_ms": 0.48, "peak_velocity_ms": 0.78,
         "eccentric_s": 1.1, "rom_m": 0.51}]}
    sc = {"score": 88, "grade": "A", "validated": True}
    s = {"dots": 110.0, "e1rm": {"e1rm_kg": 150.0}}
    adv = {"vloss": 4.0, "consistency": 92.0, "peak_drift_cm": None, "sticking_pct": None}
    path = app._session_csv(a, sc, s, adv, "Ann", 130, "vid")
    text = open(path, encoding="utf-8").read()
    assert "lifter" in text and "Ann" in text          # summary block
    assert "rep,concentric_s" in text and "0.5" in text  # per-rep table


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
