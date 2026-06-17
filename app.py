"""Gradio web app: upload a lift video -> dashboard + saved history. Local-first (Phase 2).

Run:  .\.venv\Scripts\python.exe app.py   then open http://127.0.0.1:7860
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import gradio as gr
import numpy as np

from src import advanced_metrics as am
from src import barbell, charts, confidence as conf, history, marking, media, pipeline, plate_dataset

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

# Pose backend: YOLO (GPU) locally; MediaPipe (CPU) on Hugging Face Spaces (SPACE_ID is set there).
# Override either way with the POSE_BACKEND env var.
POSE_BACKEND = os.environ.get("POSE_BACKEND") or ("mediapipe" if os.environ.get("SPACE_ID") else "yolo")

# Two-tap plate-marking prompts (the seed both steers the tracker and is saved as training data).
SEED_INSTR = "**Tap the centre** of the plate, then **tap its edge** to set the size."
SEED_INSTR_EDGE = "Centre set - now **tap the edge** of the plate to size the circle."
SEED_INSTR_DONE = "Plate set. Tap the **centre** again to redo, or press **Analyse**."

# Athletic-blue, system-font theme; follows the device's light/dark setting automatically.
THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="blue",
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
/* Mobile: edge-to-edge, full-width blocks, smaller stat numbers, scrollable table */
@media (max-width: 600px) {
  .gradio-container {max-width: 100% !important; padding: 0 8px !important;}
  .fl-narrow, .fl-video-wrap {max-width: 100% !important;}
  .fl-value {font-size: 22px;}
  .fl-narrow .table-wrap {overflow-x: auto;}
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
                         v.get("peak_velocity_ms"), v.get("eccentric_s"), v.get("rom_m")])
        else:
            rows.append([i, v.get("concentric_s"), v.get("mean_velocity_px_s"),
                         v.get("peak_velocity_px_s"), v.get("eccentric_s"), v.get("rom_px")])
    return rows


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
        return ("<div class='fl-hint'>Enter bodyweight (Settings) and a bar load to unlock "
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
    """Auto-detect the plate (largest round, saturated blob); fall back to a centred circle.
    ``frame`` is RGB. Returns (cx, cy, r, detected)."""
    cx, cy, r = w // 2, h // 2, round(h * 0.12)
    if frame is None:
        return cx, cy, r, False
    try:
        mr, mxr = int(h * 0.05), int(h * 0.25)
        hit = barbell._detect_plate(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                    np.array([np.nan, np.nan]), mr, mxr, np.pi * mr * mr * 0.5)
        if hit is not None:
            return int(hit[0][0]), int(hit[0][1]), int(hit[1]), True
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
def analyze(video_path, lift, bodyweight, sex, bar_load, cx, cy, radius, frame0, skel,
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

    progress(0.1, desc="Loading video...")
    try:
        result = pipeline.analyze(
            video_path, lift=lift, out_dir=OUT_DIR, seed=seed_tuple, backend=POSE_BACKEND,
            skeleton={"Side points": "side", "All points": "full", "None": "off"}.get(skel, "side"),
            bar_load=bar_load,
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
        _verdict_html(a, c),
        _cards_html(a, adv),
        _strength_html(s),
        charts.angle_curve(a),
        charts.velocity_time(a),
        report_md,
        _velocity_table(a),
        charts.velocity_bars(a.get("bar_velocity") or []),
        charts.bar_path(a),
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
        ax.plot(range(len(series)), [v for _, v in series], marker="o", color="#378ADD")
        ax.set_xticks(range(len(series)))
        ax.set_xticklabels([t[5:10] for t, _ in series], rotation=45, fontsize=8)
        ax.set_ylabel(metric)
        fig.tight_layout()
    return table, fig


# ---------------------------------------------------------------- UI

with gr.Blocks(title="Form Lab") as demo:
    gr.Markdown("# Form Lab")
    with gr.Tab("Analyse"):
        # --- inputs: one clean centred column (mobile-first; scales to desktop) ---
        with gr.Column(elem_classes="fl-narrow"):
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
            with gr.Row():
                lift_in = gr.Radio(["squat", "deadlift"], value="squat", label="Lift")
                load_in = gr.Number(label="Bar load (kg)", value=None)
            skel_in = gr.Radio(["Side points", "All points", "None"], value="Side points",
                               label="Skeleton", info="Side = side-view joints · None = bar path only")
            run_btn = gr.Button("Analyse", variant="primary", size="lg")

        # --- results: verdict banner, centred video, then the stat-card grid (auto-reflows) ---
        verdict_out = gr.HTML(elem_classes="fl-narrow")
        with gr.Column(elem_classes="fl-video-wrap"):
            video_out = gr.Video(label="Annotated", show_label=False, autoplay=True,
                                 elem_id="fl-video")
            gr.HTML("<div class='fl-cap'>bar speed: blue slow → red fast</div>")
        gr.HTML("<div class='fl-sec'>PER-REP VELOCITY</div>")
        reps_table = gr.Dataframe(headers=["Rep", "Con s", "Vel m/s", "Peak m/s", "Ecc s", "ROM m"],
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

        frame0_state = gr.State(None)   # clean first frame (RGB) for redraws + training save
        source_state = gr.State(None)   # full transcoded clip (trim source, non-cumulative)
        tap_state = gr.State(0)         # two-tap marker: 0 = next tap sets centre, 1 = sets radius
    with gr.Tab("History") as history_tab:
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
                  [video_in, lift_in, bw_in, sex_in, load_in, seed_cx, seed_cy, seed_radius,
                   frame0_state, skel_in],
                  [video_out, verdict_out, cards_out, strength_out, angle_out, vel_out, report_out,
                   reps_table, mcv_out, path_out])
    refresh_btn.click(load_history, [metric_in, hist_lift], [hist_table, trend_out])
    history_tab.select(load_history, [metric_in, hist_lift], [hist_table, trend_out])

if __name__ == "__main__":
    history.init_db(DB_PATH)
    demo.launch(theme=THEME, css=CSS)
