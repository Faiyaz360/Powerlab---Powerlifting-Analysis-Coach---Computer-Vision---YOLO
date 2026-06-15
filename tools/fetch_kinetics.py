"""Download just the powerlifting classes from Kinetics-700 (research/educational use).

Kinetics is a big list of YouTube clips, each with a label and a 10s start/end window. This grabs
the annotation CSV, keeps only the lifting labels, and yt-dlp-downloads each clip trimmed to its
window. Many Kinetics links are dead (videos removed over the years) — those are skipped.

Gives you hundreds of real squat/deadlift/bench clips at VARIED angles — great for pose/rep
variety and segmentation testing. (For valid form *metrics* you still need side-on clips; varied
angles are fine for training the detector and stress-testing pose — see DATA.md.)

License: the annotation list is CC BY (Google); individual videos keep their uploader's copyright —
fine for a research project, revisit before any commercial use.

Needs:  pip install yt-dlp   and ffmpeg on PATH (for trimming).
Note: yt-dlp may warn it wants a JS runtime (e.g. deno) for YouTube — install one if downloads fail
(https://github.com/yt-dlp/yt-dlp/wiki/EJS).

Usage:
    python tools/fetch_kinetics.py --split val --max-per-class 15 --out clips_kinetics
"""
from __future__ import annotations

import argparse
import csv
import io
import shutil
import subprocess
import urllib.request
from collections import defaultdict
from pathlib import Path

# Kinetics-700 labels relevant to powerlifting/weightlifting
LIFT_CLASSES = {"deadlifting", "squat", "bench pressing", "clean and jerk", "snatch weight lifting"}
CSV_URL = "https://s3.amazonaws.com/kinetics/700_2020/annotations/{split}.csv"


def _load_rows(split: str) -> list[dict]:
    url = CSV_URL.format(split=split)
    print(f"fetching annotation list: {url}")
    with urllib.request.urlopen(url, timeout=120) as r:  # noqa: S310 (fixed, trusted URL)
        text = r.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def _download_segment(yid: str, start: float, end: float, target: Path) -> bool:
    """yt-dlp-download just the [start, end] window of a YouTube video. Returns True on success."""
    cmd = [
        "yt-dlp",
        "--download-sections", f"*{start}-{end}",
        "--force-keyframes-at-cuts",
        "-f", "mp4/best",
        "--no-playlist",
        "-o", str(target),
        f"https://www.youtube.com/watch?v={yid}",
    ]
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return target.exists()


def main() -> None:
    ap = argparse.ArgumentParser(description="Download powerlifting clips from Kinetics-700 (research use)")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--max-per-class", type=int, default=15, help="cap clips per lift class")
    ap.add_argument("--out", default="clips_kinetics", help="output directory")
    args = ap.parse_args()

    if shutil.which("yt-dlp") is None:
        print("yt-dlp not found.  pip install yt-dlp")
        return
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found (needed to trim clips).")
        return

    rows = _load_rows(args.split)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("label") in LIFT_CLASSES:
            by_class[r["label"]].append(r)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for label, items in sorted(by_class.items()):
        folder = out / label.replace(" ", "_")
        folder.mkdir(exist_ok=True)
        got = 0
        for r in items:
            if got >= args.max_per_class:
                break
            yid = r["youtube_id"]
            start, end = float(r["time_start"]), float(r["time_end"])
            target = folder / f"{yid}_{int(start)}.mp4"
            if target.exists() or _download_segment(yid, start, end, target):
                got += 1
                total += 1
        print(f"{label}: {got} clips ({len(items)} listed)")
    print(f"\nDone. {total} Kinetics clips in {out}/ (research use; see DATA.md).")


if __name__ == "__main__":
    main()
