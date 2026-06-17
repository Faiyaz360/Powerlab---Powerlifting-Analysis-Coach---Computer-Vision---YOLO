"""Matplotlib figures for the Gradio dashboard. Agg backend (writes no window)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import cv2  # noqa: E402


def _fig_to_bgr(fig):
    """Render a matplotlib figure to a BGR array and close it (for on-video overlays)."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    return bgr


def _cell(x):
    if x is None:
        return "-"
    return f"{x:.2f}" if isinstance(x, float) else str(x)


def velocity_table_img(bar_velocity, width_px: int = 200, made=None):
    """Compact per-rep table (Rep | Vel | OK) as a dark BGR image for the on-video overlay — rep #,
    mean velocity, and a depth/lockout ✓/✗ (from ``made``). None if no data."""
    rows, oks = [], []
    for i, v in enumerate(bar_velocity or [], start=1):
        if not v:
            continue
        # label by the TRUE rep number (i) and align the depth/lockout flag by rep index (i-1), so a
        # skipped/None rep never shifts the numbering or the ✓/✗ onto the wrong rep.
        ok = made[i - 1] if (made is not None and i - 1 < len(made)) else None
        rows.append([str(i), _cell(v.get("mean_velocity_ms")),
                     "✓" if ok else ("✗" if ok is False else "-")])
        oks.append(ok)
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(2.2, 0.26 * (len(rows) + 1)), dpi=100)
    fig.patch.set_facecolor("#16181d")
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=["Rep", "Vel", "OK"], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#333333")
        cell.set_facecolor("#1e2128" if r == 0 else "#16181d")
        col = "#ffffff" if r == 0 else "#cfcfcf"
        if r > 0 and c == 2:                       # OK column: green tick / red cross
            col = "#5ed47a" if oks[r - 1] else ("#e06a6a" if oks[r - 1] is False else "#cfcfcf")
        cell.set_text_props(color=col)
    fig.tight_layout(pad=0.15)
    img = _fig_to_bgr(fig)
    h0, w0 = img.shape[:2]
    return cv2.resize(img, (width_px, int(h0 * width_px / w0)))


def angle_curve(analysis: dict):
    """Primary joint angle over time, with a dashed line at each rep bottom."""
    key = analysis["primary_key"]
    series = analysis["series"][key]
    t = np.arange(len(series)) / analysis["fps"]
    fig, ax = plt.subplots(figsize=(4, 2.4))
    ax.plot(t, series, color="#378ADD")
    for r in analysis.get("reps", []):
        ax.axvline(r["bottom"] / analysis["fps"], color="#E24B4A", ls="--", alpha=0.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(f"{key} angle (deg)")
    fig.tight_layout()
    return fig


def velocity_bars(bar_velocity: list):
    """Per-rep mean concentric velocity (m/s) bars with the best rep marked and velocity loss
    annotated — the coach-standard set summary. Empty-safe."""
    mcvs = [v["mean_velocity_ms"] for v in bar_velocity
            if v and v.get("mean_velocity_ms") is not None]
    fig, ax = plt.subplots(figsize=(4, 2.4))
    if not mcvs:
        ax.text(0.5, 0.5, "no bar-speed data", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    best, last = max(mcvs), mcvs[-1]
    vloss = (best - last) / best * 100 if best > 0 else 0.0
    colors = ["#1D9E75" if m == best else "#378ADD" for m in mcvs]   # best rep green, rest blue
    ax.bar([f"R{i + 1}" for i in range(len(mcvs))], mcvs, color=colors)
    ax.axhline(best, color="#1D9E75", ls="--", lw=0.9, alpha=0.6)    # best-rep reference line
    ax.set_ylabel("mean vel (m/s)")
    ax.set_title(f"velocity loss {vloss:.0f}%  (best {best:.2f} → last {last:.2f} m/s)", fontsize=8)
    fig.tight_layout()
    return fig


def velocity_time(analysis: dict):
    """Per-frame bar velocity over the whole set (upward positive) — the 'real-time' velocity trace.
    Concentric (upward) phases are shaded; rep bottoms are dashed. Empty-safe."""
    vs = analysis.get("bar_velocity_series")
    fig, ax = plt.subplots(figsize=(4, 2.4))
    if vs is None or len(vs) == 0 or not np.any(np.isfinite(vs)):
        ax.text(0.5, 0.5, "no bar-speed data", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    vs = np.asarray(vs, dtype=float)
    t = np.arange(len(vs)) / analysis["fps"]
    unit = "m/s" if analysis.get("scale_m_per_px") else "px/s"
    ax.axhline(0, color="#9AA0A6", lw=0.8)
    ax.fill_between(t, vs, 0, where=vs > 0, color="#1D9E75", alpha=0.25)  # concentric (upward)
    ax.plot(t, vs, color="#378ADD", lw=1.4)
    for r in analysis.get("bar_reps", []):
        ax.axvline(r["bottom"] / analysis["fps"], color="#E24B4A", ls="--", alpha=0.4)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(f"bar velocity ({unit})")
    fig.tight_layout()
    return fig


def bar_path(analysis: dict):
    """The bar's 2D trajectory: sideways drift (cm) vs height (cm) — the classic bar-path plot.
    A near-vertical line = a clean path; the dashed line is the start (centre). Empty-safe."""
    bar_xy = analysis.get("bar_xy")
    scale = analysis.get("scale_m_per_px")
    fig, ax = plt.subplots(figsize=(3.2, 3.4))
    if bar_xy is None or scale is None:
        ax.text(0.5, 0.5, "no bar-path data", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    x = bar_xy[:, 0].astype(float)
    y = bar_xy[:, 1].astype(float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) == 0:
        ax.text(0.5, 0.5, "no bar-path data", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    dx = (x - x[0]) * scale * 100            # sideways drift from start (cm)
    dh = (y[0] - y) * scale * 100            # height vs start (cm), up positive
    ax.plot(dx, dh, color="#378ADD", lw=1)
    ax.axvline(0, color="#9AA0A6", ls="--", lw=0.8)   # the 'centre' (start X)
    ax.set_xlabel("drift (cm)")
    ax.set_ylabel("height (cm)")
    ax.set_aspect("equal", "box")
    fig.tight_layout()
    return fig


def drift_curve(bar_xy: np.ndarray, scale_m_per_px: float | None, start: int, end: int):
    """Horizontal bar-path drift over a rep (cm). Empty-safe when uncalibrated."""
    fig, ax = plt.subplots(figsize=(4, 2.4))
    if scale_m_per_px is None or bar_xy is None:
        ax.text(0.5, 0.5, "no bar-path data", ha="center", va="center")
        ax.axis("off")
    else:
        xs = bar_xy[start:end + 1, 0].astype(float)
        drift_cm = (xs - xs[0]) * scale_m_per_px * 100
        ax.plot(np.arange(len(drift_cm)), drift_cm, color="#BA7517")
        ax.set_xlabel("frame")
        ax.set_ylabel("drift (cm)")
    fig.tight_layout()
    return fig
