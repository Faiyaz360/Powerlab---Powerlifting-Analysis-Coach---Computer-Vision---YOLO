"""Cut long lifting videos into per-set clips (so we don't scan 20 min of resting/walking).

How it works: run a fast pose model over the video and watch hip height. A set of reps shows up
as a burst where the hips bounce up and down; resting/walking is flat. We mark the active bursts,
merge ones split only by a short rest into one set, and ffmpeg-snips each set into its own clip.

The plate detector is NOT needed here — we segment from body motion alone — so this runs BEFORE
labeling, turning long videos into the short single-set clips the rest of the pipeline wants.

Needs: ultralytics, opencv-python, numpy, and ffmpeg on PATH.

Usage:
    python tools/segment_clips.py raw/long_session.mp4 [more.mp4 ...] --out clips
    python tools/segment_clips.py --in-dir raw --out clips
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

# ---- tunables (defaults work; loosen MOVE_THRESH if it misses sets) ----
SCAN_MODEL = "yolo11n-pose.pt"  # nano = fast scanner; we only need rough motion, not precision
STRIDE = 5          # run pose every Nth frame (speed)
IMGSZ = 384         # small input = fast
WIN_S = 1.0         # window (seconds) over which we measure hip travel
MOVE_THRESH = 0.25  # hip travel as a fraction of torso length; above this = "lifting"
MAX_GAP_S = 4.0     # merge two bursts separated by less rest than this into one set
MIN_SET_S = 2.0     # drop blips shorter than this (not a real set)
PAD_S = 1.0         # seconds of padding added to each side of a set
VID_EXT = (".mp4", ".mov", ".mkv", ".webm", ".avi")

# COCO keypoint indices used by YOLO pose
L_HIP, R_HIP, L_SH, R_SH = 11, 12, 5, 6


def _hip_signal(video: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Sample hip height every STRIDE frames. Returns (times_s, hip_scaled, fps).

    hip_scaled = hip-y in pixels / torso length, so motion is comparable regardless of how big the
    lifter looks in frame. NaN where no person is detected.
    """
    from ultralytics import YOLO  # imported lazily so importing this module stays light (tests)

    model = YOLO(SCAN_MODEL)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    times, hips, torsos = [], [], []
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if f % STRIDE == 0:
            res = model.predict(frame, imgsz=IMGSZ, verbose=False)[0]
            hip_y = torso = np.nan
            kp = res.keypoints
            if kp is not None and kp.conf is not None and len(kp.conf):
                p = int(kp.conf.sum(dim=1).argmax())  # most-confident person
                pts = kp.xy[p].cpu().numpy()
                hy = (pts[L_HIP, 1] + pts[R_HIP, 1]) / 2
                sy = (pts[L_SH, 1] + pts[R_SH, 1]) / 2
                hip_y, torso = hy, abs(hy - sy)
            times.append(f / fps)
            hips.append(hip_y)
            torsos.append(torso)
        f += 1
    cap.release()

    scale = np.nanmedian(np.array(torsos, dtype=float))
    if not np.isfinite(scale) or scale < 1:
        scale = 1.0
    return np.array(times, dtype=float), np.array(hips, dtype=float) / scale, fps


def _active_mask(times: np.ndarray, hip_scaled: np.ndarray, fps: float) -> np.ndarray:
    """True where hips travel more than MOVE_THRESH (of torso) within a WIN_S window."""
    n = len(hip_scaled)
    win = max(1, int(WIN_S * fps / STRIDE))
    active = np.zeros(n, dtype=bool)
    for i in range(n):
        seg = hip_scaled[max(0, i - win // 2): i + win // 2 + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size >= 2 and (seg.max() - seg.min()) > MOVE_THRESH:
            active[i] = True
    return active


def _active_intervals(
    active: np.ndarray, times: np.ndarray, max_gap: float, min_len: float, pad: float, end_t: float
) -> list[tuple[float, float]]:
    """Turn a boolean active-mask into padded (start, end) second intervals (one per merged set).

    Pure function (no video/model) so it can be unit-tested.
    """
    runs: list[list[float]] = []
    i, n = 0, len(active)
    while i < n:
        if active[i]:
            j = i
            while j + 1 < n and active[j + 1]:
                j += 1
            runs.append([float(times[i]), float(times[j])])
            i = j + 1
        else:
            i += 1
    if not runs:
        return []

    merged = [runs[0]]
    for s, e in runs[1:]:
        if s - merged[-1][1] < max_gap:  # short rest -> same set
            merged[-1][1] = e
        else:
            merged.append([s, e])

    out: list[tuple[float, float]] = []
    for s, e in merged:
        if e - s < min_len:  # too short to be a real set
            continue
        out.append((max(0.0, s - pad), min(end_t, e + pad)))
    return out


def _cut(video: Path, start: float, end: float, out_path: Path) -> None:
    """ffmpeg-snip [start, end] into out_path. -c copy = instant, no re-encode/quality loss."""
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start:.2f}", "-i", str(video),
         "-t", f"{end - start:.2f}", "-c", "copy", str(out_path)],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def segment(video: Path, out_dir: Path) -> list[dict]:
    """Scan one video, cut each detected set into its own clip, return log rows."""
    times, hip_scaled, fps = _hip_signal(video)
    if len(times) == 0:
        return []
    active = _active_mask(times, hip_scaled, fps)
    end_t = float(times[-1]) + STRIDE / fps
    intervals = _active_intervals(active, times, MAX_GAP_S, MIN_SET_S, PAD_S, end_t)

    rows = []
    for k, (s, e) in enumerate(intervals, 1):
        out_path = out_dir / f"{video.stem}_set{k:02d}.mp4"
        _cut(video, s, e, out_path)
        rows.append(
            {"source": video.name, "clip": out_path.name,
             "start_s": round(s, 2), "end_s": round(e, 2), "dur_s": round(e - s, 2)}
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Cut long lifting videos into per-set clips")
    ap.add_argument("videos", nargs="*", help="video files to segment")
    ap.add_argument("--in-dir", help="folder of videos to segment (instead of listing files)")
    ap.add_argument("--out", default="clips", help="output folder for the per-set clips")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found. Install it first (it does the actual cutting).")
        return

    vids = [Path(v) for v in args.videos]
    if args.in_dir:
        vids += [p for p in Path(args.in_dir).glob("*") if p.suffix.lower() in VID_EXT]
    if not vids:
        print("No videos given. Pass files or --in-dir <folder>.")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for v in vids:
        print(f"scanning {v.name} ...")
        rows = segment(v, out_dir)
        print(f"  -> {len(rows)} sets")
        for r in rows:  # one-line review log per cut clip
            print(f"     {r['clip']:<28} {r['start_s']:>7.2f}-{r['end_s']:<7.2f}s  ({r['dur_s']:.1f}s)")
        all_rows += rows

    if all_rows:
        with (out_dir / "segments.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["source", "clip", "start_s", "end_s", "dur_s"])
            w.writeheader()
            w.writerows(all_rows)
    print(f"\nDone. {len(all_rows)} per-set clips in {out_dir}/ (log: segments.csv)")


if __name__ == "__main__":
    main()
