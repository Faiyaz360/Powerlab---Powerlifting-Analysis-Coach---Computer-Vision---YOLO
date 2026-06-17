"""Write the outputs: metrics.json, a primary-angle plot, and a human-readable report.md."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no GUI; just write image files
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .faults import ISSUE_META  # noqa: E402

_LIFT_TITLE = {"squat": "Squat", "deadlift": "Deadlift"}


def write_report(out_dir, name, pose, analysis, faults, cues) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_path = _write_angle_plot(out_dir, name, pose, analysis)
    metrics_path = _write_metrics_json(out_dir, name, analysis, faults)
    report_path = _write_markdown(out_dir, name, analysis, faults, cues, plot_path)

    return {"report": str(report_path), "metrics": str(metrics_path), "plot": str(plot_path)}


def _write_angle_plot(out_dir, name, pose, analysis) -> Path:
    key = analysis["primary_key"]
    series = analysis["series"][key]
    t = np.arange(len(series)) / pose.fps
    plt.figure(figsize=(9, 3.5))
    plt.plot(t, series, label=f"{key} angle (deg)")
    for r in analysis["reps"]:
        marker = r["bottom"] / pose.fps
        plt.axvline(marker, color="r", linestyle="--", alpha=0.5)
    plt.xlabel("time (s)")
    plt.ylabel(f"{key} angle (deg)")
    plt.title(f"{key.capitalize()} angle over time (dashed = rep bottom)")
    plt.legend()
    plt.tight_layout()
    path = out_dir / f"{name}_{key}.png"
    plt.savefig(path, dpi=110)
    plt.close()
    return path


def _write_metrics_json(out_dir, name, analysis, faults) -> Path:
    metrics = {
        "lift": analysis["lift"],
        "rep_count": analysis["rep_count"],
        "side_analyzed": analysis["series"]["side"],
        "reps": analysis["rep_metrics"],
        "bar_velocity": analysis.get("bar_velocity"),
        "scale_m_per_px": analysis.get("scale_m_per_px"),
        "issues": faults["issue_list"],
        "tempo_cv": faults["tempo_cv"],
    }
    path = out_dir / f"{name}_metrics.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


def _write_markdown(out_dir, name, analysis, faults, cues, plot_path) -> Path:
    title = _LIFT_TITLE.get(analysis["lift"], analysis["lift"].capitalize())
    lines = [
        f"# {title} Analysis — {name}",
        "",
        f"**Reps detected:** {analysis['rep_count']}  ",
        f"**Side analyzed:** {analysis['series']['side']}",
        "",
    ]
    lines += _verdict_section(faults)
    lines += ["", "## Per-rep", ""]
    lines += _rep_table(analysis)
    lines += ["", "## Bar speed (VBT)", ""]
    lines += _bar_speed_section(analysis)
    lines += ["", "## Coaching", ""]
    lines += [f"- {c}" for c in cues]
    lines += ["", f"![{analysis['primary_key']} angle]({plot_path.name})", ""]

    path = out_dir / f"{name}_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _bar_speed_section(analysis) -> list:
    vel = analysis.get("bar_velocity") or []
    scale = analysis.get("scale_m_per_px")
    if not any(v for v in vel):
        return ["_No bar-speed data (bar anchor not tracked)._"]

    if scale:
        rows = ["| Rep | Mean vel (m/s) | Peak vel (m/s) | ROM (m) | Concentric s |",
                "|-----|----------------|----------------|---------|--------------|"]
        for i, v in enumerate(vel, 1):
            if not v:
                continue
            rows.append(f"| {i} | {v['mean_velocity_ms']} | {v['peak_velocity_ms']} | "
                        f"{v['rom_m']} | {v['concentric_s']} |")
        rows.append("")
        rows.append(f"_Calibrated from the 450 mm plate ({round(0.450 / scale)} px diameter)._")
        return rows

    rows = ["_Uncalibrated (no plate detected) — speeds in pixels/sec._", "",
            "| Rep | Mean vel (px/s) | Peak vel (px/s) | ROM (px) | Concentric s |",
            "|-----|-----------------|-----------------|----------|--------------|"]
    for i, v in enumerate(vel, 1):
        if not v:
            continue
        rows.append(f"| {i} | {v['mean_velocity_px_s']} | {v['peak_velocity_px_s']} | "
                    f"{v['rom_px']} | {v['concentric_s']} |")
    return rows


def _verdict_section(faults) -> list:
    """Competition (IPF) verdict: would the lift pass, based on legal red-light faults."""
    legal = faults["legal_issues"]
    if not legal:
        return ["## Competition verdict (IPF)", "", "✅ No red-light faults detected."]
    lines = ["## Competition verdict (IPF)", "", "❌ Would not pass — red-light fault(s):", ""]
    for issue in legal:
        rule = ISSUE_META.get(issue, {}).get("rule", "")
        lines.append(f"- **{issue.replace('_', ' ')}** — {rule}")
    return lines


def _rep_table(analysis):
    if analysis["lift"] == "deadlift":
        rows = ["| Rep | Lean° | Knee° | Lockout | Hip-rise | Ascent s |",
                "|-----|-------|-------|---------|----------|----------|"]
        for i, r in enumerate(analysis["rep_metrics"], 1):
            lock = "OK" if r["lockout_pass"] else "INCOMPLETE"
            ratio = "-" if r["hip_rise_ratio"] is None else r["hip_rise_ratio"]
            rows.append(
                f"| {i} | {r['torso_lean_deg']} | {r['lockout_knee_angle']} | {lock} | "
                f"{ratio} | {r['ascent_s']} |"
            )
        return rows

    rows = ["| Rep | Min knee° | Depth | Fwd lean° | Descent s | Ascent s |",
            "|-----|-----------|-------|-----------|-----------|----------|"]
    for i, r in enumerate(analysis["rep_metrics"], 1):
        depth = "OK" if r["depth_pass"] else "HIGH"
        rows.append(
            f"| {i} | {r['min_knee_angle']} | {depth} | {r['max_forward_lean']} | "
            f"{r['descent_s']} | {r['ascent_s']} |"
        )
    return rows
