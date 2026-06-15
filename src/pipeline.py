"""Orchestrates a full analysis run. Used by BOTH cli.py (local) and, later, app.py (cloud)."""
from __future__ import annotations

from pathlib import Path

from . import barbell as barmod
from . import coach as coachmod
from . import faults as faultsmod
from . import metrics as metricsmod
from . import pose as posemod
from . import render as rendermod
from . import report as reportmod


def analyze(video_path, lift: str = "squat", out_dir="output", progress=None,
            backend: str = "yolo", model_complexity: int = 2, plate_backend: str = "hsv") -> dict:
    """Run pose -> metrics -> faults -> coaching -> annotated video + report.

    ``backend``: "mediapipe" (CPU) or "yolo" (YOLO11-pose on GPU, sharper landmarks).
    ``model_complexity``: MediaPipe only — 0 lite / 1 full / 2 heavy (default).
    Returns a dict with the analysis, faults, cues, output paths, and rep_count.
    """
    video_path = Path(video_path)
    if lift not in ("squat", "deadlift"):
        raise NotImplementedError(f"Supported lifts: squat, deadlift. '{lift}' comes later.")

    name = video_path.stem
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pose = posemod.estimate_pose(
        video_path, backend=backend, model_complexity=model_complexity, progress=progress
    )
    if pose.num_frames == 0:
        raise ValueError("No frames decoded — is the video file valid?")

    analysis = metricsmod.analyze(pose, lift)
    analysis["pose_landmarks"] = pose.landmarks  # exposed for the confidence/off-axis check (Phase 2)
    faults = faultsmod.detect_faults(analysis, lift)
    cues = coachmod.coach_from_issues(faults["issue_list"], analysis["rep_count"])

    # bar tracking + velocity (VBT) — track the plate centre by colour, segment reps from the bar
    bar_xy, radii = barmod.track_plate(video_path, pose, lift, plate_backend=plate_backend)
    scale = barmod.scale_from_radii(radii)
    bar_reps = barmod.detect_bar_reps(bar_xy, pose.fps, scale)
    analysis["bar_xy"] = bar_xy
    analysis["scale_m_per_px"] = scale
    analysis["bar_reps"] = bar_reps
    analysis["bar_velocity"] = barmod.velocity_per_rep(bar_xy, bar_reps, pose.fps, scale)

    annotated = out_dir / f"{name}_annotated.mp4"
    rendermod.render_video(video_path, annotated, pose, analysis)

    paths = reportmod.write_report(out_dir, name, pose, analysis, faults, cues)
    paths["annotated_video"] = str(annotated)

    return {
        "analysis": analysis,
        "faults": faults,
        "cues": cues,
        "paths": paths,
        "rep_count": analysis["rep_count"],
    }
