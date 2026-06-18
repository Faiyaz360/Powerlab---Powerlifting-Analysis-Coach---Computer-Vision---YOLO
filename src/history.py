"""Local run history in a SQLite file (stdlib sqlite3 — no extra dependency).

Repository pattern: the rest of the app depends on these functions, not on SQL. The ``view``
column defaults to 'side' so a future front-view clip slots in without a migration.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

_COLUMNS = [
    "created_at", "video_name", "lifter_name", "lift", "view", "rep_count", "depth_pass",
    "confidence", "mean_velocity", "peak_velocity", "rom_m", "consistency", "velocity_loss",
    "sticking_pct", "bar_drift_cm", "bodyweight_kg", "bar_load_kg", "sex",
    "dots", "e1rm_kg", "peak_power_w", "est_rpe",
    "score", "grade", "validated",
    "annotated_path", "metrics_json_path", "notes",
]

# metrics allowed in trend() — whitelist prevents SQL injection via the column name
_TREND_METRICS = {
    "score", "consistency", "velocity_loss", "mean_velocity", "peak_velocity", "rom_m",
    "sticking_pct", "bar_drift_cm", "dots", "e1rm_kg", "peak_power_w", "est_rpe",
}


def _connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _sql_type(col: str) -> str:
    if col in ("rep_count", "depth_pass", "validated"):
        return "INTEGER"
    if col in ("created_at", "video_name", "lifter_name", "lift", "confidence", "sex",
               "grade", "annotated_path", "metrics_json_path", "notes"):
        return "TEXT"
    return "REAL"


def init_db(db_path: str) -> None:
    """Create the runs table if it does not exist, then add any columns missing from an older DB
    (lightweight migration — the leaderboard columns slot into a pre-existing table without a wipe)."""
    schema = (
        "CREATE TABLE IF NOT EXISTS runs (\n"
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
        "  view TEXT DEFAULT 'side',\n"
        + ",\n".join(f"  {c} {_sql_type(c)}" for c in _COLUMNS if c != "view")
        + "\n)"
    )
    with _connect(db_path) as conn:
        conn.execute(schema)
        existing = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
        for col in _COLUMNS:
            if col not in existing:
                conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {_sql_type(col)}")


def save_run(db_path: str, record: dict) -> int:
    """Insert a run summary; missing fields are stored as NULL. Returns the new row id."""
    init_db(db_path)
    cols = [c for c in _COLUMNS if c in record]
    placeholders = ", ".join("?" for _ in cols)
    values = [record[c] for c in cols]
    with _connect(db_path) as conn:
        cur = conn.execute(
            f"INSERT INTO runs ({', '.join(cols)}) VALUES ({placeholders})", values
        )
        return int(cur.lastrowid)


def list_runs(db_path: str, lift: str | None = None, lifter: str | None = None,
              limit: int | None = None) -> list[dict]:
    """Most-recent-first run summaries, optionally filtered by lift and/or lifter (case-insensitive)."""
    init_db(db_path)
    conds, params = [], []
    if lift:
        conds.append("lift = ?")
        params.append(lift)
    if lifter:
        conds.append("lifter_name = ? COLLATE NOCASE")
        params.append(lifter)
    sql = "SELECT * FROM runs" + (" WHERE " + " AND ".join(conds) if conds else "") + " ORDER BY id DESC"
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    with _connect(db_path) as conn:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def lifters(db_path: str) -> list[str]:
    """Distinct lifter names that have runs (for the History per-lifter filter)."""
    init_db(db_path)
    sql = ("SELECT DISTINCT lifter_name FROM runs WHERE lifter_name IS NOT NULL "
           "AND TRIM(lifter_name) != '' ORDER BY lifter_name COLLATE NOCASE")
    with _connect(db_path) as conn:
        return [row[0] for row in conn.execute(sql).fetchall()]


def bests(db_path: str, lifter: str | None = None, lift: str | None = None) -> dict:
    """Personal bests (PRs) over the runs: top score, heaviest lift, best est-1RM, fastest mean
    velocity, best DOTS. Optionally scoped to one lifter and/or lift."""
    init_db(db_path)
    conds, params = [], []
    if lifter:
        conds.append("lifter_name = ? COLLATE NOCASE")
        params.append(lifter)
    if lift:
        conds.append("lift = ?")
        params.append(lift)
    where = (" WHERE " + " AND ".join(conds)) if conds else ""
    sql = (f"SELECT MAX(score) score, MAX(bar_load_kg) weight, MAX(e1rm_kg) e1rm, "
           f"MAX(mean_velocity) mean_velocity, MAX(dots) dots FROM runs{where}")
    with _connect(db_path) as conn:
        row = conn.execute(sql, params).fetchone()
    return dict(row) if row else {}


def get_run(db_path: str, run_id: int) -> dict | None:
    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None


def trend(db_path: str, metric: str, lift: str | None = None,
          lifter: str | None = None) -> list[tuple]:
    """Oldest-first (created_at, value) pairs for one numeric metric, skipping NULLs. Optionally
    filtered by lift and/or lifter."""
    if metric not in _TREND_METRICS:
        raise ValueError(f"Unknown trend metric: {metric}")
    init_db(db_path)
    sql = f"SELECT created_at, {metric} FROM runs WHERE {metric} IS NOT NULL"
    params: list = []
    if lift:
        sql += " AND lift = ?"
        params.append(lift)
    if lifter:
        sql += " AND lifter_name = ? COLLATE NOCASE"
        params.append(lifter)
    sql += " ORDER BY id ASC"
    with _connect(db_path) as conn:
        return [(row["created_at"], row[metric]) for row in conn.execute(sql, params).fetchall()]


def load_velocity_points(db_path: str, lift: str | None = None,
                         lifter: str | None = None) -> list[tuple]:
    """(load_kg, mean_velocity) pairs for the load-velocity profile — runs that logged BOTH a bar load
    and a mean concentric velocity. Optionally filtered by lift and/or lifter."""
    init_db(db_path)
    sql = ("SELECT bar_load_kg, mean_velocity FROM runs "
           "WHERE bar_load_kg IS NOT NULL AND mean_velocity IS NOT NULL AND bar_load_kg > 0")
    params: list = []
    if lift:
        sql += " AND lift = ?"
        params.append(lift)
    if lifter:
        sql += " AND lifter_name = ? COLLATE NOCASE"
        params.append(lifter)
    with _connect(db_path) as conn:
        return [(row["bar_load_kg"], row["mean_velocity"])
                for row in conn.execute(sql, params).fetchall()]


def leaderboard(db_path: str, by: str = "score", lift: str | None = None,
                limit: int = 100) -> list[dict]:
    """Ranked best-per-lifter board over VALIDATED lifts only.

    ``by='score'`` ranks by each lifter's best execution score; ``by='weight'`` by their heaviest
    lift; ``by='dots'`` by DOTS (strength relative to bodyweight, sex-adjusted — pound-for-pound).
    One row per lifter (their single best for that board; SQLite returns the matching row's other
    columns alongside the MAX). Names are grouped case-insensitively. Optional ``lift`` filter.
    Each returned dict gains a 1-based ``rank``.
    """
    rank_cols = {"score": "score", "weight": "bar_load_kg", "dots": "dots"}
    if by not in rank_cols:
        raise ValueError(f"Unknown leaderboard sort: {by}")
    init_db(db_path)
    rank_col = rank_cols[by]
    # carry every display column EXCEPT the ranked one (it returns via MAX(...) AS rank_col), so each
    # board can still show the others (load + bodyweight + dots) in the row subtitle.
    base = ["lifter_name", "lift", "sex", "bodyweight_kg", "score", "bar_load_kg", "dots",
            "grade", "created_at"]
    cols = [c for c in base if c != rank_col]
    sql = (f"SELECT {', '.join(cols)}, MAX({rank_col}) AS {rank_col} FROM runs "
           "WHERE validated = 1 AND lifter_name IS NOT NULL AND TRIM(lifter_name) != '' "
           f"AND {rank_col} IS NOT NULL")
    params: list = []
    if lift:
        sql += " AND lift = ?"
        params.append(lift)
    # tie-break a draw on the ranked column by another metric (the screenshot's two 96/100 lifters):
    # Score ties break on DOTS, DOTS/Weight ties break on Score; created_at keeps it deterministic.
    tiebreak = {"score": "dots", "weight": "score", "dots": "score"}[by]
    sql += (f" GROUP BY lifter_name COLLATE NOCASE "
            f"ORDER BY {rank_col} DESC, {tiebreak} DESC, created_at ASC LIMIT ?")
    params.append(limit)
    with _connect(db_path) as conn:
        rows = [dict(row) for row in conn.execute(sql, params).fetchall()]
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return rows
