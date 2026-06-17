"""Leaderboard queries (best-per-lifter, validated-only) + the lightweight schema migration."""
import sqlite3

from src import history


def _rec(name, lift, weight, score, validated=1):
    return {
        "created_at": f"2026-06-17T0{weight % 7}:00:00", "video_name": "v", "lifter_name": name,
        "lift": lift, "rep_count": 3, "bar_load_kg": weight, "score": score, "grade": "A",
        "validated": validated, "sex": "male", "bodyweight_kg": 80, "dots": 100.0,
    }


def test_leaderboard_by_score_one_best_row_per_lifter(tmp_path):
    db = str(tmp_path / "h.db")
    history.save_run(db, _rec("Ann", "squat", 100, 80))
    history.save_run(db, _rec("Ann", "squat", 120, 92))   # Ann's better lift
    history.save_run(db, _rec("Bo", "squat", 140, 70))
    lb = history.leaderboard(db, by="score")
    assert [r["lifter_name"] for r in lb] == ["Ann", "Bo"]      # 92 > 70
    assert len(lb) == 2                                          # one row per lifter
    assert lb[0]["score"] == 92 and lb[0]["bar_load_kg"] == 120  # the best-scoring row's weight
    assert lb[0]["rank"] == 1 and lb[1]["rank"] == 2


def test_leaderboard_by_weight_ranks_by_heaviest(tmp_path):
    db = str(tmp_path / "h.db")
    history.save_run(db, _rec("Ann", "squat", 120, 92))
    history.save_run(db, _rec("Bo", "squat", 140, 70))
    lb = history.leaderboard(db, by="weight")
    assert [r["lifter_name"] for r in lb] == ["Bo", "Ann"]      # 140 > 120
    assert lb[0]["bar_load_kg"] == 140 and lb[0]["score"] == 70


def test_unvalidated_lifts_are_excluded(tmp_path):
    db = str(tmp_path / "h.db")
    history.save_run(db, _rec("Ann", "squat", 120, 92, validated=0))
    assert history.leaderboard(db, by="score") == []


def test_leaderboard_lift_filter(tmp_path):
    db = str(tmp_path / "h.db")
    history.save_run(db, _rec("Ann", "squat", 120, 92))
    history.save_run(db, _rec("Ann", "deadlift", 180, 88))
    only_dl = history.leaderboard(db, by="score", lift="deadlift")
    assert len(only_dl) == 1 and only_dl[0]["lift"] == "deadlift" and only_dl[0]["bar_load_kg"] == 180


def test_lifters_bests_and_per_lifter_filter(tmp_path):
    db = str(tmp_path / "h.db")
    history.save_run(db, _rec("Ann", "squat", 120, 90))
    history.save_run(db, _rec("Ann", "squat", 140, 85))
    history.save_run(db, _rec("Bo", "deadlift", 200, 80))
    assert history.lifters(db) == ["Ann", "Bo"]                  # distinct, name-sorted
    b = history.bests(db, lifter="Ann")
    assert b["score"] == 90 and b["weight"] == 140               # PRs across Ann's runs
    ann = history.list_runs(db, lifter="ann")                    # case-insensitive
    assert len(ann) == 2 and all(r["lifter_name"] == "Ann" for r in ann)
    assert [v for _, v in history.trend(db, "score", lifter="Ann")] == [90, 85]   # oldest-first


def test_init_db_migrates_an_old_table(tmp_path):
    """An older DB missing the leaderboard columns gets them added, no wipe."""
    db = str(tmp_path / "old.db")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE runs (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, lift TEXT)")
    con.commit()
    con.close()
    history.init_db(db)                                         # migrate
    rid = history.save_run(db, _rec("Ann", "squat", 120, 92))
    row = history.get_run(db, rid)
    assert row["lifter_name"] == "Ann" and row["score"] == 92 and row["validated"] == 1
