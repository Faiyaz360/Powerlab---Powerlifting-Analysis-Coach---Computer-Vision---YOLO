"""Gradio web app: upload a lift video -> dashboard + saved history. Local-first (Phase 2).

Run:  .\.venv\Scripts\python.exe app.py   then open http://127.0.0.1:7860
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import gradio as gr
import numpy as np

from src import advanced_metrics as am
from src import charts, confidence as conf, history, media, pipeline, plate_dataset

OUT_DIR = "output"
DB_PATH = "data/history.db"

# Two-click plate-marking prompts (the seed both steers the tracker and is saved as training data).
SEED_PROMPT = "**Step 1 — click the plate CENTRE** on the frame above."
SEED_EDGE = "**Step 2 — click the plate EDGE** (sets the size)."
SEED_DONE = "✅ **Plate marked.** Press **Analyse** — or click again to redo."


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


def _strength(a: dict, bodyweight_kg, sex, bar_load_kg):
    """Strength-tier scores; None until bodyweight + bar load are supplied."""
    if not bodyweight_kg or not bar_load_kg:
        return None
    mcv, peak = _first_velocity(a)
    return {
        "dots": am.dots(bar_load_kg, bodyweight_kg, sex),
        "e1rm": am.est_1rm(bar_load_kg, mcv, a["lift"]),
        "power": am.peak_power_w(bar_load_kg, peak),
        "rpe": am.velocity_to_rpe(mcv, a["lift"]),
    }


def _confidence(a: dict):
    """Off-axis / visibility confidence from the exposed pose landmarks, or None."""
    lm = a.get("pose_landmarks")
    if lm is None:
        return None
    return conf.assess(SimpleNamespace(landmarks=lm))


# ---------------------------------------------------------------- rendering helpers

def _fmt(value, suffix="") -> str:
    return f"{value}{suffix}" if value is not None else "—"


def _verdict_md(a: dict, c) -> str:
    """Confidence-gated verdict — never a confident wrong call when the camera is off-axis."""
    if c and not c["axis_ok"]:
        return f"### ⚠ Can't judge depth — {c['reason']}"
    if a["lift"] == "squat":
        ok = any(r.get("depth_pass") for r in a.get("rep_metrics") or [])
        label = "Good depth" if ok else "High — missed depth"
    else:
        ok = any(r.get("lockout_pass") for r in a.get("rep_metrics") or [])
        label = "Locked out" if ok else "Incomplete lockout"
    note = f" · {c['level']} confidence" if c else ""
    return f"### {'✅' if ok else '❌'} {label}{note}"


def _cards_md(a: dict, adv: dict) -> str:
    mcv, _ = _first_velocity(a)
    return (
        f"**Reps:** {a['rep_count']}  \n"
        f"**Mean velocity:** {_fmt(mcv, ' m/s')}  \n"
        f"**Consistency:** {_fmt(adv['consistency'], '%')}  \n"
        f"**Velocity loss:** {_fmt(adv['vloss'], '%')}  \n"
        f"**Peak bar drift:** {_fmt(adv['peak_drift_cm'], ' cm')}  \n"
        f"**Sticking point:** {_fmt(adv['sticking_pct'], '% of ascent')}"
    )


def _strength_md(s) -> str:
    if not s:
        return "_Enter bodyweight (Settings) and bar load to see strength scores._"
    e = s["e1rm"]
    e1 = f"{e['e1rm_kg']} kg ({e['confidence']} conf)" if e else "—"
    return (
        f"**DOTS:** {_fmt(s['dots'])}  \n"
        f"**Est. 1RM:** {e1}  \n"
        f"**Peak power:** {_fmt(s['power'], ' W')}  \n"
        f"**Est. RPE:** {_fmt(s['rpe'])}"
    )


def _summary_record(a, result, adv, name, c=None, s=None,
                    bodyweight=None, sex=None, bar_load=None) -> dict:
    mcv, peak = _first_velocity(a)
    depth_pass = (any(r.get("depth_pass") for r in (a.get("rep_metrics") or []))
                  if a["lift"] == "squat" else None)
    e1rm = s["e1rm"]["e1rm_kg"] if s and s.get("e1rm") else None
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "video_name": name, "lift": a["lift"], "rep_count": a["rep_count"],
        "depth_pass": int(depth_pass) if depth_pass is not None else None,
        "confidence": c["level"] if c else None,
        "mean_velocity": mcv, "peak_velocity": peak,
        "consistency": adv["consistency"], "velocity_loss": adv["vloss"],
        "sticking_pct": adv["sticking_pct"], "bar_drift_cm": adv["peak_drift_cm"],
        "bodyweight_kg": bodyweight, "bar_load_kg": bar_load, "sex": sex,
        "dots": s["dots"] if s else None, "e1rm_kg": e1rm,
        "peak_power_w": s["power"] if s else None,
        "est_rpe": s["rpe"] if s else None,
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


def _draw_seed(frame_rgb, seed):
    """Draw the in-progress mark (centre dot, and circle once the edge is set) on a copy."""
    img = frame_rgb.copy()
    cx, cy = seed.get("cx"), seed.get("cy")
    if cx is not None:
        cv2.circle(img, (int(cx), int(cy)), 5, (0, 255, 0), -1)
        if seed.get("r"):
            cv2.circle(img, (int(cx), int(cy)), int(seed["r"]), (0, 255, 0), 2)
    return img


# ---------------------------------------------------------------- callbacks

def on_upload(video_path):
    """Transcode to a browser-safe codec if needed, show its first frame to mark the plate, and
    size the trim sliders to the clip length."""
    if not video_path:
        return (None, None, None, {}, SEED_PROMPT, None,
                gr.update(maximum=1, value=0), gr.update(maximum=1, value=1))
    safe = media.browser_safe_video(video_path)
    frame = _first_frame_rgb(safe)
    dur = round(media.duration_s(safe), 1) or 1.0
    # video_in, seed_img, frame0_state, seed_state, seed_instr, source_state, trim_start, trim_end
    return (safe, frame, frame, {}, SEED_PROMPT, safe,
            gr.update(maximum=dur, value=0.0), gr.update(maximum=dur, value=dur))


def on_trim(source, start, end):
    """Cut the clip to [start, end] (frame-accurate) and refresh the preview + plate-mark frame."""
    if not source:
        raise gr.Error("Upload a video first.")
    if end <= start:
        raise gr.Error("Trim end must be after the start.")
    trimmed = media.trim(source, start, end)
    frame = _first_frame_rgb(trimmed)
    return trimmed, frame, frame, {}, SEED_PROMPT  # video_in, seed_img, frame0_state, seed_state, instr


def on_seed_click(seed_state, frame0, evt: gr.SelectData):
    """Two-click circle: 1st click = centre, 2nd = edge (radius). A 3rd click starts over."""
    if frame0 is None:
        return None, {}, SEED_PROMPT
    x, y = int(evt.index[0]), int(evt.index[1])
    s = dict(seed_state or {})
    if not s or s.get("r"):                       # fresh start (no centre yet, or already complete)
        s = {"cx": x, "cy": y}
        instr = SEED_EDGE
    else:                                         # have centre -> this click sets the radius
        s["r"] = float(np.hypot(x - s["cx"], y - s["cy"]))
        instr = SEED_DONE
    return _draw_seed(frame0, s), s, instr


def on_seed_reset(frame0):
    return frame0, {}, SEED_PROMPT


def analyze(video_path, lift, bodyweight, sex, bar_load, seed_state, frame0,
            progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Upload a lift video first.")
    seed = seed_state or {}
    if not seed.get("r"):
        raise gr.Error("Mark the plate first: click its centre, then its edge.")
    seed_tuple = (seed["cx"], seed["cy"], seed["r"])
    name = Path(video_path).stem

    # Save the mark as a YOLO training label (data flywheel) — best-effort, never blocks analysis.
    if frame0 is not None:
        try:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            plate_dataset.save_label(frame0, seed["cx"], seed["cy"], seed["r"], name, stamp=stamp)
        except Exception:
            pass

    progress(0.1, desc="Loading video...")
    try:
        result = pipeline.analyze(
            video_path, lift=lift, out_dir=OUT_DIR, seed=seed_tuple,
            progress=lambda f: progress(0.5, desc="Analysing frames..."),
        )
    except ValueError as exc:
        raise gr.Error(f"Couldn't read that video: {exc}")
    except NotImplementedError as exc:
        raise gr.Error(str(exc))

    a = result["analysis"]
    adv = _advanced(a)
    c = _confidence(a)
    s = _strength(a, bodyweight, sex, bar_load)
    history.save_run(DB_PATH, _summary_record(a, result, adv, name, c, s,
                                              bodyweight, sex, bar_load))
    report_md = Path(result["paths"]["report"]).read_text(encoding="utf-8")
    return (
        result["paths"]["annotated_video"],
        _verdict_md(a, c),
        _cards_md(a, adv),
        _strength_md(s),
        charts.angle_curve(a),
        charts.velocity_bars(a.get("bar_velocity") or []),
        report_md,
    )


def load_history(metric: str, lift: str):
    rows = history.list_runs(DB_PATH, lift=lift or None)
    table = [[r["created_at"], r["lift"], r["rep_count"], r.get("consistency"),
              r.get("velocity_loss")] for r in rows]
    series = history.trend(DB_PATH, metric, lift=lift or None)
    fig = charts.velocity_bars([])
    if series:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 2.6))
        ax.plot(range(len(series)), [v for _, v in series], marker="o", color="#1D9E75")
        ax.set_xticks(range(len(series)))
        ax.set_xticklabels([t[5:10] for t, _ in series], rotation=45, fontsize=8)
        ax.set_ylabel(metric)
        fig.tight_layout()
    return table, fig


# ---------------------------------------------------------------- UI

with gr.Blocks(title="Form Lab") as demo:
    gr.Markdown("# Form Lab — lift analysis")
    with gr.Tab("Analyse"):
        # --- inputs: upload + controls on the left, plate-mark on the right (side by side) ---
        with gr.Row():
            with gr.Column(scale=1):
                video_in = gr.Video(label="Upload (side-on)", height=360)
                with gr.Row():
                    trim_start = gr.Slider(0, 1, value=0, step=0.1, label="Trim start (s)")
                    trim_end = gr.Slider(0, 1, value=1, step=0.1, label="Trim end (s)")
                trim_btn = gr.Button("Apply trim", size="sm")
                lift_in = gr.Radio(["squat", "deadlift"], value="squat", label="Lift")
                load_in = gr.Number(label="Bar load (kg)", value=None)
                run_btn = gr.Button("Analyse", variant="primary", size="lg")
            with gr.Column(scale=1):
                seed_img = gr.Image(label="Mark the plate: click centre, then edge",
                                    type="numpy", interactive=True, sources=[], height=420)
                seed_instr = gr.Markdown(SEED_PROMPT)
                seed_reset = gr.Button("Reset mark", size="sm")

        # --- results: verdict, then a compact annotated video beside the stat cards ---
        verdict_out = gr.Markdown()
        with gr.Row():
            with gr.Column(scale=2):
                video_out = gr.Video(label="Annotated", autoplay=True, height=520)
                gr.Markdown("Bar path: 🔵 slow → 🔴 fast")
            with gr.Column(scale=3):
                cards_out = gr.Markdown()
                strength_out = gr.Markdown()

        # --- extra detail tucked away so the main screen stays uncluttered ---
        with gr.Accordion("Charts & full report", open=False):
            with gr.Row():
                angle_out = gr.Plot(label="Joint angle")
                vel_out = gr.Plot(label="Velocity per rep")
            report_out = gr.Markdown()

        frame0_state = gr.State(None)   # clean first frame (RGB) for redraws + training save
        seed_state = gr.State({})       # {cx, cy, r} being marked
        source_state = gr.State(None)   # full transcoded clip (trim source, non-cumulative)
    with gr.Tab("History"):
        with gr.Row():
            metric_in = gr.Dropdown(["consistency", "velocity_loss", "mean_velocity"],
                                    value="consistency", label="Trend metric")
            hist_lift = gr.Radio(["", "squat", "deadlift"], value="", label="Filter lift")
        refresh_btn = gr.Button("Refresh")
        hist_table = gr.Dataframe(headers=["date", "lift", "reps", "consistency", "vel loss"],
                                  label="Past runs")
        trend_out = gr.Plot(label="Trend")
    with gr.Tab("Settings"):
        bw_in = gr.Number(label="Bodyweight (kg)", value=80)
        sex_in = gr.Radio(["male", "female"], value="male", label="Sex (for DOTS)")
        gr.Markdown("_Bodyweight + sex feed DOTS / est-1RM / power / RPE. Set the bar load per "
                    "lift on the Analyse tab._")

    video_in.upload(on_upload, [video_in],
                    [video_in, seed_img, frame0_state, seed_state, seed_instr,
                     source_state, trim_start, trim_end])
    trim_btn.click(on_trim, [source_state, trim_start, trim_end],
                   [video_in, seed_img, frame0_state, seed_state, seed_instr])
    seed_img.select(on_seed_click, [seed_state, frame0_state],
                    [seed_img, seed_state, seed_instr])
    seed_reset.click(on_seed_reset, [frame0_state], [seed_img, seed_state, seed_instr])
    run_btn.click(analyze, [video_in, lift_in, bw_in, sex_in, load_in, seed_state, frame0_state],
                  [video_out, verdict_out, cards_out, strength_out, angle_out, vel_out, report_out])
    refresh_btn.click(load_history, [metric_in, hist_lift], [hist_table, trend_out])

if __name__ == "__main__":
    history.init_db(DB_PATH)
    demo.launch()
