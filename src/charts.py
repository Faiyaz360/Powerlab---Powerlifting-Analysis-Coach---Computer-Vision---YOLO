"""Matplotlib figures for the Gradio dashboard. Agg backend (writes no window)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


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
    """Mean concentric velocity per rep (m/s). Empty-safe."""
    mcvs = [v["mean_velocity_ms"] for v in bar_velocity
            if v and v.get("mean_velocity_ms") is not None]
    fig, ax = plt.subplots(figsize=(4, 2.4))
    if mcvs:
        ax.bar([f"R{i+1}" for i in range(len(mcvs))], mcvs, color="#1D9E75")
        ax.set_ylabel("mean vel (m/s)")
    else:
        ax.text(0.5, 0.5, "no bar-speed data", ha="center", va="center")
        ax.axis("off")
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
