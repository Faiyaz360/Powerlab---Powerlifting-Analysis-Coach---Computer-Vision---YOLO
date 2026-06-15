"""Headless end-to-end smoke check for the Phase 2 dashboard path (no browser).

Picks the smallest available lift clip, runs the real pipeline + dashboard helpers + history
save, and asserts the annotated video and a saved row exist. Replaces the manual browser smoke
for unattended runs.

Run:  .\.venv\Scripts\python.exe tools/smoke_dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app  # noqa: E402  (app exposes the dashboard helpers)
from src import history, pipeline  # noqa: E402

SMOKE_DB = "data/smoke_history.db"
SEARCH_DIRS = ["input", "clips_kinetics/squat", "clips_kinetics/deadlifting"]
VIDEO_EXTS = {".mov", ".mp4", ".avi", ".m4v"}


def _find_smallest_clip():
    clips = []
    for d in SEARCH_DIRS:
        p = ROOT / d
        if p.is_dir():
            clips += [f for f in p.iterdir()
                      if f.suffix.lower() in VIDEO_EXTS and f.stat().st_size > 0]
    return min(clips, key=lambda f: f.stat().st_size) if clips else None


def main() -> int:
    clip = _find_smallest_clip()
    if clip is None:
        print("SKIP: no input clip found in", SEARCH_DIRS)
        return 0
    lift = "deadlift" if "deadlift" in str(clip).lower() else "squat"
    print(f"smoke clip: {clip.name} ({clip.stat().st_size} bytes), lift={lift}")

    # mediapipe backend = CPU only, no GPU/torch (safe + fast for a smoke check)
    result = pipeline.analyze(str(clip), lift=lift, out_dir="output",
                              backend="mediapipe", model_complexity=1, plate_backend="hsv")
    a = result["analysis"]
    adv = app._advanced(a)
    rec = app._summary_record(a, result, adv, clip.stem)
    rid = history.save_run(SMOKE_DB, rec)
    saved = history.get_run(SMOKE_DB, rid)

    annotated = Path(result["paths"]["annotated_video"])
    assert annotated.exists(), f"annotated video missing: {annotated}"
    assert saved is not None and saved["video_name"] == clip.stem, "history row not saved"

    print("rep_count:", a["rep_count"], "| advanced:", adv)
    print("annotated:", annotated, "exists:", annotated.exists())
    print("history row id:", rid, "| view:", saved["view"])
    print("SMOKE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
