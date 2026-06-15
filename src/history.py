"""Local run history in a SQLite file (stdlib sqlite3 — no extra dependency).

Repository pattern: the rest of the app depends on these functions, not on SQL. The ``view``
column defaults to 'side' so a future front-view clip slots in without a migration.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

_COLUMNS = [
    "created_at", "video_name", "lift", "view", "rep_count", "depth_pass", "confidence",
    "mean_velocity", "peak_velocity", "rom_m", "consistency", "velocity_loss",
    "sticking_pct", "bar_drift_cm", "bodyweight_kg", "bar_load_kg", "sex",
    "dots", "e1rm_kg", "peak_power_w", "est_rpe",
    "annotated_path", "metrics_json_path", "notes",
]

# metrics allowed in trend() — whitelist prevents SQL injection via the column name
_TREND_METRICS = {
    "consistency", "velocity_loss", "mean_velocity", "peak_velocity", "rom_m",
    "sticking_pct", "bar_drift_cm", "dots", "e1rm_kg", "peak_power_w", "est_rpe",
}


def _connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _sql_type(col: str) -> str:
    if col in ("rep_count", "depth_pass"):
        return "INTEGER"
    if col in ("created_at", "video_name", "lift", "confidence", "sex",
               "annotated_path", "metrics_json_path", "notes"):
        return "TEXT"
    return "REAL"


def init_db(db_path: str) -> None:
    """Create the runs table if it does not exist."""
    schema = (
        "CREATE TABLE IF NOT EXISTS runs (\n"
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
        "  view TEXT DEFAULT 'side',\n"
        + ",\n".join(f"  {c} {_sql_type(c)}" for c in _COLUMNS if c != "view")
        + "\n)"
    )
    with _connect(db_path) as conn:
        conn.execute(schema)


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


def list_runs(db_path: str, lift: str | None = None, limit: int | None = None) -> list[dict]:
    """Most-recent-first run summaries, optionally filtered by lift."""
    init_db(db_path)
    sql = "SELECT * FROM runs"
    params: list = []
    if lift:
        sql += " WHERE lift = ?"
        params.append(lift)
    sql += " ORDER BY id DESC"
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    with _connect(db_path) as conn:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def get_run(db_path: str, run_id: int) -> dict | None:
    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None


def trend(db_path: str, metric: str, lift: str | None = None) -> list[tuple]:
    """Oldest-first (created_at, value) pairs for one numeric metric, skipping NULLs."""
    if metric not in _TREND_METRICS:
        raise ValueError(f"Unknown trend metric: {metric}")
    init_db(db_path)
    sql = f"SELECT created_at, {metric} FROM runs WHERE {metric} IS NOT NULL"
    params: list = []
    if lift:
        sql += " AND lift = ?"
        params.append(lift)
    sql += " ORDER BY id ASC"
    with _connect(db_path) as conn:
        return [(row["created_at"], row[metric]) for row in conn.execute(sql, params).fetchall()]
