"""Make uploaded videos browser-playable.

Phone clips are often HEVC / H.265 (and 10-bit) — Chrome can't decode those, so the in-app
preview shows a blank video with only audio. We transcode just those to H.264 8-bit for the web
UI. Dimensions are unchanged, so a click on the preview still maps 1:1 to the analysis frames.

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


def _needs_transcode(codec: str | None, pix_fmt: str | None) -> bool:
    """Whether a browser likely can't play this stream (pure decision — unit-tested)."""
    if codec is None:
        return False  # unknown — don't touch it, let the browser try
    if codec not in _BROWSER_SAFE_CODECS:
        return True
    if pix_fmt is not None and pix_fmt not in _BROWSER_SAFE_PIXFMTS:
        return True  # e.g. 10-bit H.264 still won't play in Chrome
    return False


def _probe(path: Path):
    """Return (codec_name, pix_fmt) of the first video stream, or (None, None) if unknown."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,pix_fmt",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30,
        ).stdout.split()
    except Exception:
        return None, None
    codec = out[0] if len(out) >= 1 else None
    pix_fmt = out[1] if len(out) >= 2 else None
    return codec, pix_fmt


def browser_safe_video(path, out_dir="output/_preview") -> str:
    """Return a path to a browser-playable copy, transcoding (H.264 8-bit) only if needed."""
    path = Path(path)
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        return str(path)
    codec, pix_fmt = _probe(path)
    if not _needs_transcode(codec, pix_fmt):
        return str(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{path.stem}_h264.mp4"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23",
             "-c:a", "aac", "-movflags", "+faststart", str(out)],
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
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23",
             "-c:a", "aac", "-movflags", "+faststart", str(out)],
            capture_output=True, timeout=300, check=True,
        )
    except Exception:
        return str(path)
    return str(out)
