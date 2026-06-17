"""Gradio web app: upload a lift video -> dashboard + saved history. Local-first (Phase 2).

Run:  .\.venv\Scripts\python.exe app.py   then open http://127.0.0.1:7860
"""
from __future__ import annotations

import csv
import os
import shutil
from datetime import datetime
from html import escape
from pathlib import Path
from types import SimpleNamespace

import cv2
import gradio as gr
import numpy as np

from src import advanced_metrics as am
from src import barbell, charts, confidence as conf, history, marking, media, pipeline
from src import plate_dataset, score

try:
    import spaces  # Hugging Face ZeroGPU — allocates a GPU for the decorated call
except ImportError:  # local / non-ZeroGPU: make the decorator a harmless no-op
    class spaces:  # noqa: N801
        @staticmethod
        def GPU(*_args, **_kwargs):
            def _wrap(fn):
                return fn
            return _wrap

OUT_DIR = "output"
DB_PATH = "data/history.db"

# Persistence: HF Spaces' disk is ephemeral, so the leaderboard DB is snapshotted to a mounted HF
# Storage Bucket (whole-file copy — bucket-friendly, no live-SQLite-on-object-store risk). Auto-on
# when a bucket is mounted at /data (or PERSIST_DIR is set); a no-op locally where /data doesn't exist.
PERSIST_DIR = os.environ.get("PERSIST_DIR") or (
    "/data" if os.path.isdir("/data") and os.access("/data", os.W_OK) else None)


def _restore_db() -> None:
    """On boot, copy the persisted leaderboard DB from the bucket into the working path (if present)."""
    if not PERSIST_DIR:
        return
    saved = os.path.join(PERSIST_DIR, "history.db")
    if os.path.exists(saved):
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(saved, DB_PATH)


def _snapshot_db() -> None:
    """Copy the working DB out to the mounted bucket after a save (best-effort; never blocks)."""
    if not PERSIST_DIR or not os.path.exists(DB_PATH):
        return
    try:
        os.makedirs(PERSIST_DIR, exist_ok=True)
        shutil.copy2(DB_PATH, os.path.join(PERSIST_DIR, "history.db"))
    except Exception:
        pass

# Pose backend: YOLO (GPU) locally; MediaPipe (CPU) on Hugging Face Spaces (SPACE_ID is set there).
# Override either way with the POSE_BACKEND env var.
POSE_BACKEND = os.environ.get("POSE_BACKEND") or ("mediapipe" if os.environ.get("SPACE_ID") else "yolo")

# Two-tap plate-marking prompts (the seed both steers the tracker and is saved as training data).
SEED_INSTR = "**Tap the centre** of the plate, then **tap its edge** to set the size."
SEED_INSTR_EDGE = "Centre set - now **tap the edge** of the plate to size the circle."
SEED_INSTR_DONE = "Plate set. Tap the **centre** again to redo, or press **Analyse**."

# Black-glass, faint-purple theme (the dark palette + glass surfaces are layered in CSS below).
THEME = gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="purple",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_lg,
    font=["-apple-system", "BlinkMacSystemFont", "SF Pro Text", "Segoe UI", "Roboto", "sans-serif"],
)

# Apple-clean polish: a responsive stat-card grid, a semantic verdict banner, centred video.
CSS = """
footer {display: none !important;}
.gradio-container {max-width: 920px !important; margin: 0 auto !important; min-height: 100vh;}
.fl-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px;}
.fl-card {background: var(--block-background-fill); border: 1px solid var(--border-color-primary);
          border-radius: 16px; padding: 14px 16px;}
.fl-label {display: block; font-size: 12px; color: var(--body-text-color-subdued); margin-bottom: 4px;}
.fl-value {font-size: 25px; font-weight: 600; color: var(--body-text-color); line-height: 1.1;}
.fl-unit {font-size: 13px; color: var(--body-text-color-subdued); font-weight: 400;}
.fl-verdict {border-radius: 16px; padding: 14px 16px; display: flex; align-items: center; gap: 12px;
             font-size: 17px; font-weight: 600; margin: 4px 0;}
.fl-verdict .fl-vicon {font-size: 22px; line-height: 1;}
.fl-sub {display: block; font-size: 12px; font-weight: 400; opacity: .8; margin-top: 2px;}
.fl-ok {background: rgba(34,197,94,.13); color: #16a34a;}
.fl-warn {background: rgba(245,158,11,.15); color: #d97706;}
.fl-bad {background: rgba(239,68,68,.13); color: #dc2626;}
.fl-hint {color: var(--body-text-color-subdued); font-size: 14px; padding: 8px 2px;}
.fl-video-wrap {max-width: 460px; margin: 0 auto;}
.fl-cap {text-align: center; font-size: 12px; color: var(--body-text-color-subdued); margin-top: 6px;}
.fl-sec {font-size: 13px; font-weight: 600; color: var(--body-text-color-subdued);
         margin: 14px 0 6px; letter-spacing: .02em;}
.fl-narrow {max-width: 560px; margin: 0 auto;}
#fl-video, #fl-video video {background: #0b0b0c !important; border-radius: 14px;}
#fl-video video {width: 100%; height: auto;}   /* fill width, keep aspect (portrait fills the phone) */
#fl-seed, #fl-seed img {max-height: 70vh; object-fit: contain;}  /* tall marking frame stays on-screen */
.fl-narrow .table-wrap {border-radius: 12px;}
/* score banner */
.fl-score {display: flex; align-items: center; gap: 16px; background: var(--block-background-fill);
           border: 1px solid var(--border-color-primary); border-radius: 16px; padding: 16px; margin: 4px 0;}
.fl-score-num {font-size: 44px; font-weight: 700; line-height: 1; color: var(--body-text-color);}
.fl-score-num .fl-unit {font-size: 16px;}
.fl-grade {font-size: 22px; font-weight: 700; padding: 4px 12px; border-radius: 12px;
           background: rgba(37,138,221,.15); color: #2b82dd;}
.fl-score-meta {flex: 1;}
.fl-score-status {font-size: 13px; color: var(--body-text-color-subdued); margin-top: 3px;}
.fl-bars {display: grid; grid-template-columns: repeat(auto-fit, minmax(86px, 1fr)); gap: 6px; margin-top: 10px;}
.fl-bar {font-size: 11px; color: var(--body-text-color-subdued);}
.fl-bar i {display: block; height: 6px; border-radius: 3px; background: rgba(37,138,221,.18); margin-top: 3px;}
.fl-bar i b {display: block; height: 100%; background: #2b82dd; border-radius: 3px;}
/* AI coach panel — prominent, right under the video */
.fl-coach {background: var(--block-background-fill); border: 1px solid var(--border-color-primary);
           border-left: 4px solid #f5a623; border-radius: 16px; padding: 16px 20px; margin: 4px 0;}
.fl-coach p {margin: 0 0 10px; line-height: 1.55; font-size: 15px; color: var(--body-text-color);}
.fl-coach p:first-child {font-weight: 600;}   /* the opener / compliment pops */
.fl-coach p:last-child {margin-bottom: 0;}
/* leaderboard */
.lb {display: flex; flex-direction: column; gap: 8px;}
.lb-row {display: flex; align-items: center; gap: 12px; padding: 12px 14px; border-radius: 14px;
         background: var(--block-background-fill); border: 1px solid var(--border-color-primary);}
.lb-medal {font-size: 20px; width: 36px; text-align: center; font-weight: 700;
           color: var(--body-text-color-subdued);}
.lb-name {flex: 1; font-weight: 600; font-size: 16px; color: var(--body-text-color);}
.lb-sub {display: block; font-weight: 400; font-size: 12px; color: var(--body-text-color-subdued);}
.lb-grade {font-size: 13px; font-weight: 700; color: #2b82dd; min-width: 26px; text-align: center;}
.lb-primary {font-size: 22px; font-weight: 700; color: var(--body-text-color);}
.lb-primary .fl-unit {font-size: 13px;}
.lb-rank1 {border-color: rgba(245,197,24,.55); background: linear-gradient(0deg, rgba(245,197,24,.10), transparent);}
.lb-rank2 {border-color: rgba(184,192,200,.55);}
.lb-rank3 {border-color: rgba(205,127,50,.50);}
/* Mobile: edge-to-edge, full-width blocks, smaller stat numbers, scrollable table */
@media (max-width: 600px) {
  .gradio-container {max-width: 100% !important; padding: 0 8px !important;}
  .fl-narrow, .fl-video-wrap {max-width: 100% !important;}
  .fl-value {font-size: 22px;}
  .fl-narrow .table-wrap {overflow-x: auto;}
  .fl-score-num {font-size: 36px;}
}

/* ===== Black-glass + faint-purple theme (mobile-first, easy on the eyes) ===== */
:root, .gradio-container, .gradio-container.dark, .dark {
  --body-background-fill: #0a0a0f;
  --background-fill-primary: rgba(255,255,255,.04);
  --background-fill-secondary: rgba(255,255,255,.06);
  --block-background-fill: rgba(255,255,255,.045);
  --block-border-color: rgba(150,130,255,.14);
  --border-color-primary: rgba(150,130,255,.16);
  --body-text-color: rgba(237,235,247,.92);
  --body-text-color-subdued: rgba(198,194,218,.58);
  --block-label-text-color: rgba(198,194,218,.72);
  --input-background-fill: rgba(255,255,255,.05);
  --neutral-950: #0a0a0f; --neutral-900: #0d0d14;
}
body, .gradio-container {
  background: radial-gradient(1100px 560px at 50% -10%, rgba(124,92,255,.12), transparent 62%), #0a0a0f !important;
  color: var(--body-text-color);
}
.gradio-container {max-width: 680px !important;}   /* focused, phone-shaped even on desktop */
/* glass surfaces */
.fl-card, .fl-score, .fl-coach, .fl-verdict, .lb-row, .block {
  background: rgba(255,255,255,.045) !important;
  border: 1px solid rgba(150,130,255,.14) !important;
  -webkit-backdrop-filter: blur(16px) saturate(1.25);
  backdrop-filter: blur(16px) saturate(1.25);
}
/* sections: airy, quiet labels */
.fl-sec {margin: 24px 0 8px !important; font-size: 12px; letter-spacing: .06em; text-transform: uppercase;
         color: rgba(198,194,218,.5) !important;}
.fl-value, .fl-score-num {color: #f3f1fb !important;}   /* crisp but not harsh white */
/* purple accents (replacing the old blue) */
.fl-grade {background: rgba(124,92,255,.2) !important; color: #c9bdff !important;}
.fl-bar i {background: rgba(124,92,255,.16) !important;}
.fl-bar i b {background: #a78bfa !important;}
.fl-coach {border-left: 3px solid #a78bfa !important;}
.lb-grade {color: #c9bdff !important;}
.lb-rank1 {border-color: rgba(124,92,255,.45) !important;
           background: linear-gradient(0deg, rgba(124,92,255,.12), transparent) !important;}
/* verdict colours — brighter on dark */
.fl-ok {background: rgba(52,211,153,.14) !important; color: #6ee7b7 !important;}
.fl-warn {background: rgba(245,158,11,.16) !important; color: #fcd34d !important;}
.fl-bad {background: rgba(248,113,113,.15) !important; color: #fca5a5 !important;}
/* primary button in purple */
button.primary, .gr-button-primary, .primary {
  background: linear-gradient(180deg, #8b6cff, #6f4dff) !important;
  border: 1px solid rgba(150,130,255,.45) !important; color: #fff !important;}
/* generous mobile spacing */
@media (max-width: 600px) {
  .fl-card, .fl-score, .fl-coach, .fl-verdict {padding: 16px 18px !important;}
  .fl-sec {margin: 20px 0 7px !important;}
}
"""


# ---------------------------------------------------------------- metric helpers

def _consistency_features(a: dict) -> dict:
    """Per-rep features used for the consistency score (real keys from metrics.py)."""
    rm = a.get("rep_metrics") or []
    bv = a.get("bar_velocity") or []
    depth_key = "min_knee_angle" if a["lift"] == "squat" else "lockout_hip_angle"
    return {
        "depth": [r.get(depth_key) for r in rm],
        "ascent": [r.get("ascent_s") for r in rm],
        "mcv": [v.get("mean_velocity_ms") for v in bv if v],
    }


def _advanced(a: dict) -> dict:
    """Compute the four free metrics from the analysis dict."""
    bv = a.get("bar_velocity") or []
    bar_xy = a.get("bar_xy")
    scale = a.get("scale_m_per_px")
    reps = a.get("bar_reps") or []
    drifts, sticks = [], []
    if bar_xy is not None:
        bar_y = bar_xy[:, 1]
        for r in reps:
            d = am.bar_path_drift(bar_xy, scale, r["bottom"], r["top"])
            if d:
                drifts.append(d["peak_drift_cm"])
            s = am.sticking_point_pct(bar_y, r["bottom"], r["top"])
            if s:
                sticks.append(s["pct_of_rom"])
    return {
        "vloss": am.velocity_loss_pct(bv),
        "consistency": am.consistency_score(_consistency_features(a)),
        "peak_drift_cm": round(max(drifts), 1) if drifts else None,
        "sticking_pct": round(float(np.median(sticks)), 0) if sticks else None,
    }


def _first_velocity(a: dict):
    """Mean/peak concentric velocity of the first tracked rep, or (None, None)."""
    bv = [v for v in (a.get("bar_velocity") or []) if v]
    if not bv:
        return None, None
    return bv[0].get("mean_velocity_ms"), bv[0].get("peak_velocity_ms")


def _velocity_table(a: dict):
    """Per-rep rows for the stats table under the video (LIFT-APP style):
    Rep | Con(s) | Vel(m/s) | Peak(m/s) | Ecc(s) | ROM(m). Calibrated when the plate was marked
    (always, here); else raw px units."""
    rows = []
    for i, v in enumerate(a.get("bar_velocity") or [], start=1):
        if not v:
            continue
        if v.get("calibrated"):
            rows.append([i, v.get("concentric_s"), v.get("mean_velocity_ms"),
                         v.get("peak_velocity_ms"), v.get("eccentric_s"), v.get("rom_m"),
                         am.velocity_zone(v.get("mean_velocity_ms")) or "—"])
        else:
            rows.append([i, v.get("concentric_s"), v.get("mean_velocity_px_s"),
                         v.get("peak_velocity_px_s"), v.get("eccentric_s"), v.get("rom_px"), "—"])
    return rows


def _session_csv(a, sc, s, adv, lifter_name, bar_load, name) -> str:
    """Write a spreadsheet-friendly CSV of the session (a summary block + per-rep rows) to OUT_DIR;
    return its path for a download button. Open in Sheets/Excel to track lifts over time."""
    mcv, _ = _first_velocity(a)
    e1rm = s["e1rm"]["e1rm_kg"] if s and s.get("e1rm") else ""
    summary = [
        ("lifter", lifter_name or ""), ("lift", a["lift"]), ("lift_weight_kg", bar_load or ""),
        ("score_/100", sc["score"] if sc else ""), ("grade", sc["grade"] if sc else ""),
        ("validated", sc["validated"] if sc else ""), ("reps", a["rep_count"]),
        ("mean_velocity_ms", mcv if mcv is not None else ""),
        ("velocity_loss_pct", adv["vloss"] if adv["vloss"] is not None else ""),
        ("consistency_pct", adv["consistency"] if adv["consistency"] is not None else ""),
        ("dots", s["dots"] if s else ""), ("est_1rm_kg", e1rm),
    ]
    path = os.path.join(OUT_DIR, f"{name}_session.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerows(summary)
        w.writerow([])
        w.writerow(["rep", "concentric_s", "mean_vel_ms", "peak_vel_ms", "eccentric_s", "rom_m", "zone"])
        for i, v in enumerate(a.get("bar_velocity") or [], start=1):
            if not v:
                continue
            w.writerow([i, v.get("concentric_s"), v.get("mean_velocity_ms"),
                        v.get("peak_velocity_ms"), v.get("eccentric_s"), v.get("rom_m"),
                        am.velocity_zone(v.get("mean_velocity_ms")) or ""])
    return path


def _strength(a: dict, bodyweight_kg, sex, bar_load_kg):
    """Strength-tier scores; None until bodyweight + bar load are supplied."""
    if not bodyweight_kg or not bar_load_kg:
        return None
    _, peak = _first_velocity(a)
    bv = [v for v in (a.get("bar_velocity") or []) if v]
    reps = len(bv)
    last_mcv = bv[-1].get("mean_velocity_ms") if bv else None
    return {
        "dots": am.dots(bar_load_kg, bodyweight_kg, sex),
        "e1rm": am.est_1rm(bar_load_kg, reps, last_mcv, a["lift"]),
        "power": am.peak_power_w(bar_load_kg, peak),
        "rpe": am.velocity_to_rpe(last_mcv, a["lift"]),
    }


def _confidence(a: dict):
    """Off-axis / visibility confidence from the exposed pose landmarks, or None."""
    lm = a.get("pose_landmarks")
    if lm is None:
        return None
    return conf.assess(SimpleNamespace(landmarks=lm))


# ---------------------------------------------------------------- HTML rendering (stat cards)

def _card(label, value, unit="") -> str:
    """One stat card: muted label above, big number below (— when the value is missing)."""
    if value is None:
        inner = "<span class='fl-value'>—</span>"
    else:
        u = f"<span class='fl-unit'> {unit}</span>" if unit else ""
        inner = f"<span class='fl-value'>{value}{u}</span>"
    return f"<div class='fl-card'><span class='fl-label'>{label}</span>{inner}</div>"


def _verdict_html(a: dict, c) -> str:
    """Confidence-gated verdict banner — never a confident wrong call when the camera is off-axis."""
    if c and not c["axis_ok"]:
        what = "depth" if a["lift"] == "squat" else "lockout"
        return (f"<div class='fl-verdict fl-warn'><span class='fl-vicon'>⚠</span>"
                f"<div>Can't judge {what}<span class='fl-sub'>{c['reason']}</span></div></div>")
    if a["lift"] == "squat":
        ok = any(r.get("depth_pass") for r in a.get("rep_metrics") or [])
        label = "Good depth" if ok else "High — missed depth"
    else:
        ok = any(r.get("lockout_pass") for r in a.get("rep_metrics") or [])
        label = "Locked out" if ok else "Incomplete lockout"
    cls, icon = ("fl-ok", "✓") if ok else ("fl-bad", "✗")
    sub = f"{c['level']} confidence" if c else ""
    return (f"<div class='fl-verdict {cls}'><span class='fl-vicon'>{icon}</span>"
            f"<div>{label}<span class='fl-sub'>{sub}</span></div></div>")


def _cards_html(a: dict, adv: dict) -> str:
    mcv, _ = _first_velocity(a)
    rm = a.get("rep_metrics") or []
    if a["lift"] == "squat":                       # lift-specific primary metric (no depth on deadlifts)
        made = sum(1 for r in rm if r.get("depth_pass"))
        primary = _card("Depth made", f"{made}/{len(rm)}" if rm else None)
    else:
        made = sum(1 for r in rm if r.get("lockout_pass"))
        primary = _card("Lockouts", f"{made}/{len(rm)}" if rm else None)
    cards = [
        _card("Reps", a["rep_count"]),
        primary,
        _card("Mean velocity", mcv, "m/s"),
        _card("Consistency", adv["consistency"], "%"),
        _card("Velocity loss", adv["vloss"], "%"),
        _card("Peak bar drift", adv["peak_drift_cm"], "cm"),
        _card("Sticking point", adv["sticking_pct"], "% asc"),
    ]
    return f"<div class='fl-grid'>{''.join(cards)}</div>"


def _strength_html(s) -> str:
    if not s:
        return ("<div class='fl-hint'>Enter bodyweight and a lift weight above to unlock "
                "strength scores.</div>")
    e = s["e1rm"]
    e1_val = e["e1rm_kg"] if e else None
    cards = [
        _card("DOTS", s["dots"]),
        _card("Est. 1RM", e1_val, f"kg · {e['confidence']}" if e else "kg"),
        _card("Peak power", s["power"], "W"),
        _card("Est. RPE", s["rpe"]),
    ]
    hint = ("<div class='fl-hint'>Est. 1RM = your reps + how close the last rep was to failure "
            "(from its bar speed). Most accurate when you take the set near failure; it can't see "
            "reps you didn't do.</div>")
    return f"<div class='fl-grid'>{''.join(cards)}</div>{hint}"


_SCORE_BARS = [("Legal", "legality"), ("Technique", "technique"), ("Bar path", "bar_path"),
               ("Control", "control"), ("Consistency", "consistency")]


def _coaching_html(cues) -> str:
    """Prominent AI-coaching panel shown right under the video (never hidden in an accordion).

    Renders the coach's lines — opener (compliment), the fix cues, then the forward-looking close —
    as a stacked card so it reads like a coach talking between sets.
    """
    if not cues:
        return ""
    lines = "".join(f"<p>{escape(c)}</p>" for c in cues)
    return f"<div class='fl-coach'>{lines}</div>"


def _score_html(sc) -> str:
    """Gamified execution-score banner: big /100, letter grade, leaderboard status, component bars."""
    if not sc:
        return ""
    status = ("\U0001F3C5 Validated — on the leaderboard" if sc["validated"]
              else "Not on the leaderboard — " + sc["reason"].split("—", 1)[-1].strip())
    bars = "".join(
        f"<span class='fl-bar'>{lbl}<i><b style='width:{int(sc['breakdown'][k])}%'></b></i></span>"
        for lbl, k in _SCORE_BARS if sc["breakdown"].get(k) is not None
    )
    return (f"<div class='fl-score'>"
            f"<span class='fl-score-num'>{sc['score']}<span class='fl-unit'>/100</span></span>"
            f"<span class='fl-grade'>{escape(str(sc['grade']))}</span>"
            f"<div class='fl-score-meta'><b>Lift score</b>"
            f"<div class='fl-score-status'>{escape(status)}</div>"
            f"<div class='fl-bars'>{bars}</div></div></div>")


_MEDALS = {1: "\U0001F947", 2: "\U0001F948", 3: "\U0001F949"}   # gold / silver / bronze


def _leaderboard_html(rows: list, by: str) -> str:
    """Ranked board. ``by`` = 'Score' or 'Weight'. Names are user input -> HTML-escaped."""
    if not rows:
        return ("<div class='fl-hint'>No validated lifts yet. Analyse a <b>side-on</b> lift with your "
                "<b>name</b> and <b>lift weight</b> filled in, and pass depth/lockout, to claim a spot.</div>")
    items = []
    for r in rows:
        rank = r["rank"]
        medal = _MEDALS.get(rank, f"#{rank}")
        weight = r.get("bar_load_kg")
        sc_val = r.get("score")
        if by == "Score":
            primary = f"{sc_val:.0f}<span class='fl-unit'>/100</span>" if sc_val is not None else "—"
            sub = f"{weight:.0f} kg · {escape(str(r.get('lift', '')))}" if weight else escape(str(r.get("lift", "")))
        else:
            primary = f"{weight:.0f}<span class='fl-unit'> kg</span>" if weight else "—"
            sub = f"score {sc_val:.0f} · {escape(str(r.get('lift', '')))}" if sc_val is not None else escape(str(r.get("lift", "")))
        dots = f" · DOTS {r['dots']:.0f}" if r.get("dots") else ""
        grade = escape(str(r.get("grade") or ""))
        items.append(
            f"<div class='lb-row lb-rank{min(rank, 4)}'>"
            f"<span class='lb-medal'>{medal}</span>"
            f"<span class='lb-name'>{escape(str(r.get('lifter_name', '')))}"
            f"<span class='lb-sub'>{sub}{dots}</span></span>"
            f"<span class='lb-grade'>{grade}</span>"
            f"<span class='lb-primary'>{primary}</span></div>"
        )
    return f"<div class='lb'>{''.join(items)}</div>"


def load_board(by: str, lift: str):
    """Render the leaderboard ranked by score or by weight, optionally filtered to one lift."""
    rows = history.leaderboard(DB_PATH, by="score" if by == "Score" else "weight",
                               lift=lift or None, limit=100)
    return _leaderboard_html(rows, by)


def _summary_record(a, result, adv, name, c=None, s=None,
                    bodyweight=None, sex=None, bar_load=None,
                    lifter_name=None, sc=None, validated=0) -> dict:
    mcv, peak = _first_velocity(a)
    depth_pass = (any(r.get("depth_pass") for r in (a.get("rep_metrics") or []))
                  if a["lift"] == "squat" else None)
    e1rm = s["e1rm"]["e1rm_kg"] if s and s.get("e1rm") else None
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "video_name": name, "lifter_name": lifter_name, "lift": a["lift"],
        "rep_count": a["rep_count"],
        "depth_pass": int(depth_pass) if depth_pass is not None else None,
        "confidence": c["level"] if c else None,
        "mean_velocity": mcv, "peak_velocity": peak,
        "consistency": adv["consistency"], "velocity_loss": adv["vloss"],
        "sticking_pct": adv["sticking_pct"], "bar_drift_cm": adv["peak_drift_cm"],
        "bodyweight_kg": bodyweight, "bar_load_kg": bar_load, "sex": sex,
        "dots": s["dots"] if s else None, "e1rm_kg": e1rm,
        "peak_power_w": s["power"] if s else None,
        "est_rpe": s["rpe"] if s else None,
        "score": sc["score"] if sc else None, "grade": sc["grade"] if sc else None,
        "validated": int(validated),
        "annotated_path": result["paths"]["annotated_video"],
        "metrics_json_path": result["paths"]["metrics"],
    }


# ---------------------------------------------------------------- seed (plate marking) helpers

def _first_frame_rgb(video_path):
    """First frame of the clip as an RGB array (for the click-to-mark picker), or None."""
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None


def _draw_reticle(frame_rgb, cx, cy, r):
    """Semi-transparent blue disk + outline + white crosshair — the targeting reticle the user
    lines up with the plate (centre crosshair + sized circle)."""
    r = max(1, int(r))
    img = frame_rgb.copy()
    overlay = img.copy()
    cv2.circle(overlay, (cx, cy), r, (37, 138, 221), -1)
    img = cv2.addWeighted(overlay, 0.22, img, 0.78, 0)
    cv2.circle(img, (cx, cy), r, (37, 138, 221), 2)
    cv2.line(img, (cx - 14, cy), (cx + 14, cy), (255, 255, 255), 1)
    cv2.line(img, (cx, cy - 14), (cx, cy + 14), (255, 255, 255), 1)
    cv2.circle(img, (cx, cy), 3, (255, 255, 255), -1)
    return img


def _reticle_view(frame0, cx, cy, r):
    """Frame with the reticle drawn at (cx, cy, r), or None if there's no frame yet."""
    if frame0 is None:
        return None
    return _draw_reticle(frame0, int(cx), int(cy), int(r))


# ---------------------------------------------------------------- callbacks

def on_upload(video_path):
    """Transcode if needed, show the first frame with a centred reticle, and size every slider."""
    if not video_path:
        return (None, None, None, SEED_INSTR, None,
                gr.update(maximum=1, value=0), gr.update(maximum=1, value=1),
                gr.update(), gr.update(), gr.update(), 0)
    safe = media.browser_safe_video(video_path)
    frame = _first_frame_rgb(safe)
    dur = round(media.duration_s(safe), 1) or 1.0
    h, w = (frame.shape[0], frame.shape[1]) if frame is not None else (480, 640)
    cx, cy, r, detected = _auto_or_default(frame, h, w)   # auto-detect the plate on upload
    instr = ("Auto-detected the plate ✓ — check it's on the plate (tap to fix), then **Analyse**."
             if detected else SEED_INSTR)
    # ...seed_instr, source_state, trim_start, trim_end, cx, cy, radius, tap_state
    return (safe, _reticle_view(frame, cx, cy, r), frame, instr, safe,
            gr.update(maximum=dur, value=0.0), gr.update(maximum=dur, value=dur),
            gr.update(maximum=w, value=cx), gr.update(maximum=h, value=cy),
            gr.update(maximum=max(20, h // 2), value=r), 0)


def on_trim(source, start, end, cx, cy, r):
    """Cut the clip to [start, end] (frame-accurate) and redraw the reticle on the new frame."""
    if not source:
        raise gr.Error("Upload a video first.")
    if end <= start:
        raise gr.Error("Trim end must be after the start.")
    trimmed = media.trim(source, start, end)
    frame = _first_frame_rgb(trimmed)
    return trimmed, _reticle_view(frame, cx, cy, r), frame  # video_in, seed_img, frame0_state


def on_reticle(frame0, cx, cy, r):
    """Redraw the reticle as the centre / radius sliders move."""
    return _reticle_view(frame0, cx, cy, r)


def on_tap(frame0, cx, cy, r, tap_state, evt: gr.SelectData):
    """Two-tap plate marking (touch-native): the first tap drops the centre, the second sets the
    radius from how far it lands. Redraws the reticle, updates the sliders, and advances the
    prompt for the next tap."""
    if frame0 is None or evt.index is None:
        return (gr.update(),) * 6
    x, y = int(evt.index[0]), int(evt.index[1])
    cx, cy, r, nxt = marking.tap_to_seed(tap_state, x, y, int(cx), int(cy), int(r))
    instr = SEED_INSTR_EDGE if nxt == 1 else SEED_INSTR_DONE
    return _reticle_view(frame0, cx, cy, r), cx, cy, r, nxt, instr


def _auto_or_default(frame, h, w):
    """Auto-detect the plate (colour blob, then a shape/Hough fallback for dull/black plates); fall
    back to a centred circle. ``frame`` is RGB. Returns (cx, cy, r, detected)."""
    cx, cy, r = w // 2, h // 2, round(h * 0.12)
    if frame is None:
        return cx, cy, r, False
    try:
        hit = barbell.detect_plate_seed(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), h)
        if hit is not None:
            return int(hit[0]), int(hit[1]), int(hit[2]), True
    except Exception:
        pass
    return cx, cy, r, False


def on_autodetect(frame0):
    """Re-run plate auto-detection and fill the marker — adjustable before analysing. Auto-detect
    trades a little scale accuracy for speed, so it's a convenience, not a careful mark."""
    if frame0 is None:
        return gr.update(), gr.update(), gr.update(), gr.update(), 0, SEED_INSTR
    h, w = frame0.shape[:2]
    cx, cy, r, detected = _auto_or_default(frame0, h, w)
    instr = ("Auto-detected the plate ✓ — adjust if needed, then **Analyse**." if detected
             else "No plate found automatically — tap the plate to mark it.")
    return _reticle_view(frame0, cx, cy, r), cx, cy, r, 0, instr


@spaces.GPU(duration=120)
def analyze(video_path, lifter_name, lift, bodyweight, sex, bar_load, cx, cy, radius, frame0, skel,
            progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Upload a lift video first.")
    if not radius or radius <= 0:
        raise gr.Error("Align the plate circle first (use the X / Y / radius sliders).")
    seed_tuple = (cx, cy, radius)
    name = Path(video_path).stem

    # Save the mark as a YOLO training label (data flywheel) — best-effort, never blocks analysis.
    if frame0 is not None:
        try:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            plate_dataset.save_label(frame0, cx, cy, radius, name, stamp=stamp)
        except Exception:
            pass

    name_clean = (lifter_name or "").strip()
    progress(0.1, desc="Loading video...")
    try:
        result = pipeline.analyze(
            video_path, lift=lift, out_dir=OUT_DIR, seed=seed_tuple, backend=POSE_BACKEND,
            skeleton={"Side points": "side", "All points": "full", "None": "off"}.get(skel, "side"),
            bar_load=bar_load, lifter_name=name_clean or None, sex=sex, bodyweight=bodyweight,
            progress=lambda f: progress(0.5, desc="Analysing frames..."),
        )
    except ValueError as exc:
        raise gr.Error(f"Couldn't read that video: {exc}")
    except NotImplementedError as exc:
        raise gr.Error(str(exc))

    a = result["analysis"]
    adv = _advanced(a)
    c = a.get("confidence") or _confidence(a)          # computed once in the pipeline (shared)
    s = _strength(a, bodyweight, sex, bar_load)
    sc = a.get("lift_score") or score.score_lift(a, result["faults"], c)
    # A lift reaches the leaderboard only when it's validated (side-on + legal) AND attributable
    # (a name) with a weight to rank by.
    on_board = bool(sc and sc["validated"] and name_clean and bar_load)
    history.save_run(DB_PATH, _summary_record(a, result, adv, name, c, s, bodyweight, sex, bar_load,
                                              lifter_name=name_clean or None, sc=sc,
                                              validated=int(on_board)))
    _snapshot_db()   # persist the leaderboard to the mounted bucket (no-op if none)
    csv_path = _session_csv(a, sc, s, adv, name_clean, bar_load, name)
    report_md = Path(result["paths"]["report"]).read_text(encoding="utf-8")
    return (
        result["paths"]["annotated_video"],
        _coaching_html(result["cues"]),
        _verdict_html(a, c),
        _score_html(sc),
        _cards_html(a, adv),
        _strength_html(s),
        charts.angle_curve(a),
        charts.velocity_time(a),
        report_md,
        _velocity_table(a),
        charts.velocity_bars(a.get("bar_velocity") or []),
        charts.bar_path(a),
        csv_path,
    )


def _trend_fig(series, metric):
    """Line chart of one metric over time (oldest -> newest). Empty-safe placeholder when no data."""
    fig = charts.velocity_bars([])
    if series:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 2.6))
        ax.plot(range(len(series)), [v for _, v in series], marker="o", color="#378ADD")
        ax.set_xticks(range(len(series)))
        ax.set_xticklabels([t[5:10] for t, _ in series], rotation=45, fontsize=8)
        ax.set_ylabel(metric)
        fig.tight_layout()
    return fig


def _bests_html(b) -> str:
    """Personal-best cards (PRs) — top score, heaviest, best e1RM, fastest mean velocity, best DOTS."""
    if not b or all(v is None for v in b.values()):
        return "<div class='fl-hint'>No saved lifts yet — analyse a few to see your bests here.</div>"
    cards = [
        _card("Best score", f"{b['score']:g}" if b.get("score") is not None else None, "/100"),
        _card("Heaviest", f"{b['weight']:g}" if b.get("weight") is not None else None, "kg"),
        _card("Best est. 1RM", f"{b['e1rm']:g}" if b.get("e1rm") is not None else None, "kg"),
        _card("Top mean vel", f"{b['mean_velocity']:g}" if b.get("mean_velocity") is not None else None, "m/s"),
        _card("Best DOTS", f"{b['dots']:g}" if b.get("dots") is not None else None),
    ]
    return f"<div class='fl-grid'>{''.join(cards)}</div>"


def load_history(metric: str, lift: str, lifter: str):
    """Past-runs table + a metric trend + personal-best cards, filtered by lift and/or lifter."""
    lf = lifter or None
    rows = history.list_runs(DB_PATH, lift=lift or None, lifter=lf)
    table = [[r["created_at"][:16], r.get("lifter_name") or "—", r["lift"], r.get("bar_load_kg"),
              r.get("score"), r.get("e1rm_kg"), r.get("mean_velocity")] for r in rows]
    fig = _trend_fig(history.trend(DB_PATH, metric, lift=lift or None, lifter=lf), metric)
    bests = _bests_html(history.bests(DB_PATH, lifter=lf, lift=lift or None))
    return table, fig, bests


def on_history_open(metric: str, lift: str, lifter: str):
    """Tab-open: refresh the lifter dropdown choices, then load (value kept if still valid)."""
    table, fig, bests = load_history(metric, lift, lifter)
    return gr.update(choices=[""] + history.lifters(DB_PATH)), table, fig, bests


# ---------------------------------------------------------------- UI

with gr.Blocks(title="Form Lab") as demo:
    gr.Markdown("# Form Lab")
    with gr.Tab("Analyse"):
        # --- inputs: one clean centred column (mobile-first; scales to desktop) ---
        with gr.Column(elem_classes="fl-narrow"):
            # lifter details first — drive the leaderboard + strength scores
            gr.HTML("<div class='fl-sec'>LIFTER</div>")
            with gr.Row():
                name_in = gr.Textbox(label="Name", placeholder="Your name (for the leaderboard)",
                                     max_lines=1)
                sex_in = gr.Radio(["male", "female"], value="male", label="Gender")
            with gr.Row():
                bw_in = gr.Number(label="Bodyweight (kg)", value=80)
                load_in = gr.Number(label="Lift weight (kg)", value=None)
            lift_in = gr.Radio(["squat", "deadlift"], value="squat", label="Lift")

            gr.HTML("<div class='fl-sec'>VIDEO (SIDE-ON)</div>")
            video_in = gr.Video(label="Upload (side-on)", height=260)
            with gr.Accordion("Trim clip (optional)", open=False):
                with gr.Row():
                    trim_start = gr.Slider(0, 1, value=0, step=0.1, label="Start (s)")
                    trim_end = gr.Slider(0, 1, value=1, step=0.1, label="End (s)")
                trim_btn = gr.Button("Apply trim", size="sm")
            seed_img = gr.Image(label="Mark the plate — align the circle", type="numpy",
                                interactive=False, elem_id="fl-seed")
            seed_instr = gr.Markdown(SEED_INSTR)
            auto_btn = gr.Button("Auto-detect plate (quick)", size="sm")
            with gr.Accordion("Adjust by hand (optional)", open=False):
                with gr.Row():
                    seed_cx = gr.Slider(0, 1, value=0, step=1, label="Centre X")
                    seed_cy = gr.Slider(0, 1, value=0, step=1, label="Centre Y")
                seed_radius = gr.Slider(10, 300, value=60, step=1, label="Plate radius")
            skel_in = gr.Radio(["Side points", "All points", "None"], value="Side points",
                               label="Skeleton", info="Side = side-view joints · None = bar path only")
            run_btn = gr.Button("Analyse", variant="primary", size="lg")

        # --- results: verdict + score banner, centred video, then the stat-card grid (auto-reflows) ---
        verdict_out = gr.HTML(elem_classes="fl-narrow")
        score_out = gr.HTML(elem_classes="fl-narrow")
        with gr.Column(elem_classes="fl-video-wrap"):
            video_out = gr.Video(label="Annotated", show_label=False, autoplay=True,
                                 elem_id="fl-video")
            gr.HTML("<div class='fl-cap'>bar speed: blue slow → red fast</div>")
        gr.HTML("<div class='fl-sec'>AI COACH</div>")
        coach_out = gr.HTML(elem_classes="fl-narrow")
        gr.HTML("<div class='fl-sec'>PER-REP VELOCITY</div>")
        reps_table = gr.Dataframe(
            headers=["Rep", "Con s", "Vel m/s", "Peak m/s", "Ecc s", "ROM m", "Zone"],
            interactive=False, elem_classes="fl-narrow")
        mcv_out = gr.Plot(label="Mean velocity per rep", show_label=False, elem_classes="fl-narrow")
        vel_out = gr.Plot(label="Bar velocity over time", show_label=False, elem_classes="fl-narrow")
        cards_out = gr.HTML()
        gr.HTML("<div class='fl-sec'>STRENGTH</div>")
        strength_out = gr.HTML()

        # --- extra detail tucked away so the main screen stays uncluttered ---
        with gr.Accordion("Charts & full report", open=False):
            with gr.Row():
                angle_out = gr.Plot(label="Joint angle")
                path_out = gr.Plot(label="Bar path")
            report_out = gr.Markdown()
        csv_out = gr.File(label="Download session data (CSV)", elem_classes="fl-narrow")

        frame0_state = gr.State(None)   # clean first frame (RGB) for redraws + training save
        source_state = gr.State(None)   # full transcoded clip (trim source, non-cumulative)
        tap_state = gr.State(0)         # two-tap marker: 0 = next tap sets centre, 1 = sets radius
    with gr.Tab("History") as history_tab:
        with gr.Row():
            hist_lifter = gr.Dropdown([""], value="", label="Lifter", allow_custom_value=True)
            hist_lift = gr.Radio(["", "squat", "deadlift"], value="", label="Lift")
            metric_in = gr.Dropdown(
                ["score", "e1rm_kg", "mean_velocity", "peak_velocity", "dots", "consistency",
                 "velocity_loss"], value="score", label="Trend metric")
        refresh_btn = gr.Button("Refresh")
        gr.HTML("<div class='fl-sec'>PERSONAL BESTS</div>")
        bests_out = gr.HTML()
        hist_table = gr.Dataframe(
            headers=["date", "lifter", "lift", "weight", "score", "e1RM", "mean vel"],
            label="Past runs", elem_classes="fl-narrow")
        trend_out = gr.Plot(label="Trend")
    with gr.Tab("Leaderboard") as board_tab:
        gr.Markdown("🏆 **Leaderboard** — each lifter's best validated lift. "
                    "**Score** = how well you lifted (/100) · **Weight** = how much. "
                    "Only side-on, legal lifts (with a name + weight) count.")
        with gr.Row():
            board_by = gr.Radio(["Score", "Weight"], value="Score", label="Rank by")
            board_lift = gr.Radio(["", "squat", "deadlift"], value="", label="Lift")
        board_refresh = gr.Button("Refresh")
        board_out = gr.HTML(elem_classes="fl-narrow")

    video_in.upload(on_upload, [video_in],
                    [video_in, seed_img, frame0_state, seed_instr, source_state,
                     trim_start, trim_end, seed_cx, seed_cy, seed_radius, tap_state])
    trim_btn.click(on_trim, [source_state, trim_start, trim_end, seed_cx, seed_cy, seed_radius],
                   [video_in, seed_img, frame0_state])
    for _sld in (seed_cx, seed_cy, seed_radius):
        _sld.release(on_reticle, [frame0_state, seed_cx, seed_cy, seed_radius], seed_img)
    seed_img.select(on_tap, [frame0_state, seed_cx, seed_cy, seed_radius, tap_state],
                    [seed_img, seed_cx, seed_cy, seed_radius, tap_state, seed_instr])
    auto_btn.click(on_autodetect, [frame0_state],
                   [seed_img, seed_cx, seed_cy, seed_radius, tap_state, seed_instr])
    run_btn.click(analyze,
                  [video_in, name_in, lift_in, bw_in, sex_in, load_in, seed_cx, seed_cy, seed_radius,
                   frame0_state, skel_in],
                  [video_out, coach_out, verdict_out, score_out, cards_out, strength_out, angle_out, vel_out,
                   report_out, reps_table, mcv_out, path_out, csv_out],
                  show_progress_on=[video_out])   # one progress bar (on the video), not one per output
    _hist_in = [metric_in, hist_lift, hist_lifter]
    _hist_out = [hist_table, trend_out, bests_out]
    for _c in (metric_in, hist_lift, hist_lifter):
        _c.change(load_history, _hist_in, _hist_out)
    refresh_btn.click(load_history, _hist_in, _hist_out)
    history_tab.select(on_history_open, _hist_in, [hist_lifter, hist_table, trend_out, bests_out])
    board_tab.select(load_board, [board_by, board_lift], board_out)
    board_refresh.click(load_board, [board_by, board_lift], board_out)
    board_by.change(load_board, [board_by, board_lift], board_out)
    board_lift.change(load_board, [board_by, board_lift], board_out)

if __name__ == "__main__":
    _restore_db()              # pull the persisted leaderboard from the bucket (if mounted) first
    history.init_db(DB_PATH)   # then create / migrate the working DB
    demo.launch(theme=THEME, css=CSS)
