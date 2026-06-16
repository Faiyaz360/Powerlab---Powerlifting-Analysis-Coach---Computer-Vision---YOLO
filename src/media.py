"""Make uploaded videos browser-playable.

Phone clips are often HEVC / H.265 (and 10-bit, 4K) — Chrome can't decode those, and full 4K is
slow to analyse. We normalise an uploaded clip in ONE ffmpeg pass: transcode to H.264 8-bit (so the
browser can play it) AND downscale the longer side to a working size. The SAME normalised clip is
used for the preview and the analysis, so a click on the preview still maps 1:1 to the analysis
frames — and pose, tracking and rendering all run on far fewer pixels (much faster).

ffmpeg must be on PATH. If a clip is already browser-safe, it's returned untouched; if ffmpeg
is missing or the transcode fails, we fall back to the original (analysis still works — only the
in-browser preview is affected).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

# Video codecs browsers decode natively. Anything else (hevc, prores, vp9-in-mov, ...) we transcode.
_BROWSER_SAFE_CODECS = {"h264", "avc1", "vp8", "theora"}
# Even H.264 must be 8-bit 4:2:0 to play everywhere; 10-bit (yuv420p10le) will not.
_BROWSER_SAFE_PIXFMTS = {"yuv420p", "yuvj420p"}

# Cap the longer side of the working clip. Downscaling here makes the transcode, pose, bar tracking
# and rendering all faster (fewer pixels) with no real accuracy cost — the pose model resizes
# internally anyway, and the marked plate scales with the frame. 1280 keeps the plate well-resolved.
TARGET_LONG_SIDE = 960


def _needs_transcode(codec: str | None, pix_fmt: str | None) -> bool:
    """Whether a browser likely can't play this stream (pure decision — unit-tested)."""
    if codec is None:
        return False  # unknown — don't touch it, let the browser try
    if codec not in _BROWSER_SAFE_CODECS:
        return True
    if pix_fmt is not None and pix_fmt not in _BROWSER_SAFE_PIXFMTS:
        return True  # e.g. 10-bit H.264 still won't play in Chrome
    return False


def _needs_downscale(width, height) -> bool:
    """Whether the clip is bigger than our analysis working size (pure — unit-tested)."""
    if not width or not height:
        return False
    return max(width, height) > TARGET_LONG_SIDE


def _probe(path: Path):
    """Return (codec_name, pix_fmt, width, height) of the first video stream.

    Uses ``key=value`` output and parses by key, so it's robust to ffprobe's field ordering.
    Unknown fields come back as None / 0.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,pix_fmt,width,height",
             "-of", "default=noprint_wrappers=1", str(path)],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except Exception:
        return None, None, 0, 0
    d = dict(line.split("=", 1) for line in out.strip().splitlines() if "=" in line)
    w = int(d["width"]) if d.get("width", "").isdigit() else 0
    h = int(d["height"]) if d.get("height", "").isdigit() else 0
    return d.get("codec_name"), d.get("pix_fmt"), w, h


def browser_safe_video(path, out_dir="output/_preview") -> str:
    """Return a browser-playable, analysis-sized copy of the clip.

    Normalises in ONE ffmpeg pass when the clip either can't play in a browser (HEVC / 10-bit) OR is
    larger than our working size: transcode to H.264 8-bit AND downscale the longer side to
    ``TARGET_LONG_SIDE`` (audio dropped — we don't use it). Already-small, browser-safe clips are
    returned untouched. Falls back to the original on any failure (analysis still works).
    """
    path = Path(path)
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        return str(path)
    codec, pix_fmt, w, h = _probe(path)
    if not _needs_transcode(codec, pix_fmt) and not _needs_downscale(w, h):
        return str(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{path.stem}_web.mp4"
    # Fit inside a TARGET×TARGET box, keep aspect, downscale-only, even dimensions for yuv420p.
    scale = (f"scale='min({TARGET_LONG_SIDE},iw)':'min({TARGET_LONG_SIDE},ih)'"
             ":force_original_aspect_ratio=decrease:force_divisible_by=2")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-vf", scale,
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-crf", "23",
             "-an", "-movflags", "+faststart", str(out)],
            capture_output=True, timeout=300, check=True,
        )
    except Exception:
        return str(path)  # transcode failed — original still analyses fine
    return str(out)


def duration_s(path) -> float:
    """Clip length in seconds (0.0 if unknown). Used to size the trim sliders."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        return float(out) if out else 0.0
    except Exception:
        return 0.0


def trim(path, start, end, out_dir="output/_preview") -> str:
    """Cut [start, end] seconds out of the clip (re-encoded H.264, frame-accurate).

    Returns the trimmed path, or the original on any failure. Always trims from the passed source
    so re-trimming is not cumulative.
    """
    path = Path(path)
    if shutil.which("ffmpeg") is None or end <= start:
        return str(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{path.stem}_trim.mp4"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-ss", str(start), "-to", str(end),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-crf", "23",
             "-c:a", "aac", "-movflags", "+faststart", str(out)],
            capture_output=True, timeout=300, check=True,
        )
    except Exception:
        return str(path)
    return str(out)
