"""Gradio web app: upload a lift video -> dashboard + saved history. Local-first (Phase 2).

Run:  .\.venv\Scripts\python.exe app.py   then open http://127.0.0.1:7860
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np

from src import advanced_metrics as am
from src import charts, history, pipeline

OUT_DIR = "output"
DB_PATH = "data/history.db"


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


def _fmt(value, suffix="") -> str:
    return f"{value}{suffix}" if value is not None else "—"


def _cards_md(a: dict, adv: dict) -> str:
    bv = [v for v in (a.get("bar_velocity") or []) if v]
    mcv = bv[0]["mean_velocity_ms"] if bv else None
    return (
        f"**Reps:** {a['rep_count']}  \n"
        f"**Mean velocity:** {_fmt(mcv, ' m/s')}  \n"
        f"**Consistency:** {_fmt(adv['consistency'], '%')}  \n"
        f"**Velocity loss:** {_fmt(adv['vloss'], '%')}  \n"
        f"**Peak bar drift:** {_fmt(adv['peak_drift_cm'], ' cm')}  \n"
        f"**Sticking point:** {_fmt(adv['sticking_pct'], '% of ascent')}"
    )


def _summary_record(a: dict, result: dict, adv: dict, name: str) -> dict:
    bv = [v for v in (a.get("bar_velocity") or []) if v]
    depth_pass = (any(r.get("depth_pass") for r in (a.get("rep_metrics") or []))
                  if a["lift"] == "squat" else None)
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "video_name": name,
        "lift": a["lift"],
        "rep_count": a["rep_count"],
        "depth_pass": int(depth_pass) if depth_pass is not None else None,
        "mean_velocity": bv[0]["mean_velocity_ms"] if bv else None,
        "consistency": adv["consistency"],
        "velocity_loss": adv["vloss"],
        "sticking_pct": adv["sticking_pct"],
        "bar_drift_cm": adv["peak_drift_cm"],
        "annotated_path": result["paths"]["annotated_video"],
        "metrics_json_path": result["paths"]["metrics"],
    }


def analyze(video_path: str, lift: str, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Upload a lift video first.")
    progress(0.1, desc="Loading video...")
    try:
        result = pipeline.analyze(
            video_path, lift=lift, out_dir=OUT_DIR,
            progress=lambda f: progress(0.5, desc="Analysing frames..."),
        )
    except ValueError as exc:
        raise gr.Error(f"Couldn't read that video: {exc}")
    except NotImplementedError as exc:
        raise gr.Error(str(exc))

    a = result["analysis"]
    adv = _advanced(a)
    name = Path(video_path).stem
    history.save_run(DB_PATH, _summary_record(a, result, adv, name))

    report_md = Path(result["paths"]["report"]).read_text(encoding="utf-8")
    return (
        result["paths"]["annotated_video"],
        _cards_md(a, adv),
        charts.angle_curve(a),
        charts.velocity_bars(a.get("bar_velocity") or []),
        report_md,
    )


def load_history(metric: str, lift: str):
    """Return a table of past runs and a trend figure for the chosen metric."""
    rows = history.list_runs(DB_PATH, lift=lift or None)
    table = [[r["created_at"], r["lift"], r["rep_count"], r.get("consistency"),
              r.get("velocity_loss")] for r in rows]
    series = history.trend(DB_PATH, metric, lift=lift or None)
    fig = charts.velocity_bars([])  # reuse empty-safe figure if no data
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


with gr.Blocks(title="Form Lab") as demo:
    gr.Markdown("# Form Lab — lift analysis")
    with gr.Tab("Analyse"):
        with gr.Row():
            video_in = gr.Video(label="Your lift (side-on)")
            lift_in = gr.Radio(["squat", "deadlift"], value="squat", label="Lift")
        run_btn = gr.Button("Analyse", variant="primary")
        with gr.Row():
            video_out = gr.Video(label="Annotated")
            cards_out = gr.Markdown()
        with gr.Row():
            angle_out = gr.Plot(label="Joint angle")
            vel_out = gr.Plot(label="Velocity per rep")
        report_out = gr.Markdown()
        run_btn.click(analyze, [video_in, lift_in],
                      [video_out, cards_out, angle_out, vel_out, report_out])
    with gr.Tab("History"):
        with gr.Row():
            metric_in = gr.Dropdown(["consistency", "velocity_loss", "mean_velocity"],
                                    value="consistency", label="Trend metric")
            hist_lift = gr.Radio(["", "squat", "deadlift"], value="", label="Filter lift")
        refresh_btn = gr.Button("Refresh")
        hist_table = gr.Dataframe(headers=["date", "lift", "reps", "consistency", "vel loss"],
                                  label="Past runs")
        trend_out = gr.Plot(label="Trend")
        refresh_btn.click(load_history, [metric_in, hist_lift], [hist_table, trend_out])

if __name__ == "__main__":
    history.init_db(DB_PATH)
    demo.launch()
