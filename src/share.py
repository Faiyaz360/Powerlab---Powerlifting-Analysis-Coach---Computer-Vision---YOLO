"""Build a shareable social clip from an analysed lift.

Two jobs, both pure and testable:
  - ``make_share_clip`` re-frames the annotated (landscape) video into a 9:16 portrait clip
    with a blurred fill background and a small FORM LAB + score brand overlay baked in.
  - ``share_caption`` builds the post caption (tags @projectfyz, real lift numbers, hashtags).

The owner's Instagram handle is shown as a NOTE + pre-filled in the caption — it is deliberately
NOT watermarked onto the video frames.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import media as mediamod

HANDLE = "@projectfyz"            # owner's IG — tagged in the caption, not burned into the video
OUT_W, OUT_H = 1080, 1920        # 9:16 portrait (Reels / TikTok / Shorts)
PURPLE = (240, 123, 139)         # theme accent #8b7bf0, in OpenCV BGR
WHITE = (255, 255, 255)
MUTED = (200, 200, 206)
PANEL = (14, 14, 18)             # dark pill behind the score, matches the app theme
_F = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------- caption

def _kg(load) -> str:
    return f"{load:g}kg"


def share_caption(meta: dict) -> str:
    """Post caption from the lift's real numbers. Deterministic, tags @projectfyz, ready to paste."""
    lift = (meta.get("lift") or "lift").lower()
    head = f"{lift.capitalize()} @ {_kg(meta['load'])}" if meta.get("load") else lift.capitalize()
    lines = [f"{head} — analysed in Form Lab"]

    facts = []
    legal = meta.get("legal_pass")
    if legal is not None:
        word = "Depth" if lift == "squat" else "Lockout"
        facts.append(f"{word}: {'passed ✅' if legal else 'missed'}")
    if meta.get("score") is not None:
        grade = meta.get("grade")
        facts.append(f"Score {meta['score']}/100" + (f" ({grade})" if grade else ""))
    if facts:
        lines.append(" · ".join(facts))

    if meta.get("peak_ms"):
        lines.append(f"Top bar speed {meta['peak_ms']:g} m/s")

    lines.append(f"AI form breakdown → {HANDLE}")
    tag = lift if lift in ("squat", "deadlift") else "powerlifting"
    lines.append(f"#powerlifting #{tag} #formcheck #VBT #gymtok")
    return "\n".join(lines)


# ---------------------------------------------------------------- portrait clip

def _blurred_bg(frame: np.ndarray) -> np.ndarray:
    """A darkened, blurred cover-fill of the frame for the portrait letterbox area (no black bars)."""
    h, w = frame.shape[:2]
    scale = max(OUT_W / w, OUT_H / h)
    bw, bh = int(np.ceil(w * scale)), int(np.ceil(h * scale))
    # Blur cheaply on a 1/6-scale copy, then upscale — far faster than blurring at full size.
    small = cv2.resize(frame, (max(1, bw // 6), max(1, bh // 6)))
    small = cv2.GaussianBlur(small, (0, 0), 6)
    big = cv2.resize(small, (bw, bh))
    x, y = (bw - OUT_W) // 2, (bh - OUT_H) // 2
    bg = big[y:y + OUT_H, x:x + OUT_W]
    return (bg * 0.45).astype(np.uint8)


def _brand(canvas: np.ndarray, lift: str, score, grade) -> None:
    """Bake the FORM LAB wordmark (top) + the /100 score pill (bottom) onto the portrait canvas."""
    cv2.putText(canvas, "FORM", (60, 110), _F, 1.6, WHITE, 4, cv2.LINE_AA)
    (fw, _), _ = cv2.getTextSize("FORM ", _F, 1.6, 4)
    cv2.putText(canvas, "LAB", (60 + fw, 110), _F, 1.6, PURPLE, 4, cv2.LINE_AA)
    cv2.putText(canvas, (lift or "").upper(), (62, 152), _F, 0.7, MUTED, 2, cv2.LINE_AA)

    if score is not None:
        label = f"{grade}  {score}/100" if grade else f"{score}/100"
        (tw, th), _ = cv2.getTextSize(label, _F, 1.1, 3)
        x, y = 60, OUT_H - 90
        cv2.rectangle(canvas, (x - 20, y - th - 24), (x + tw + 24, y + 18), PANEL, -1)
        cv2.rectangle(canvas, (x - 20, y - th - 24), (x + tw + 24, y + 18), PURPLE, 2)
        cv2.putText(canvas, label, (x, y), _F, 1.1, WHITE, 3, cv2.LINE_AA)


def _compose(frame: np.ndarray, lift: str, score, grade) -> np.ndarray:
    """One source frame -> one branded 9:16 portrait frame (blurred fill + centred original)."""
    h, w = frame.shape[:2]
    canvas = _blurred_bg(frame)
    fg_w, fg_h = OUT_W, max(1, round(h * OUT_W / w))     # fit to width...
    if fg_h > OUT_H:                                     # ...unless that overflows height
        fg_w, fg_h = max(1, round(w * OUT_H / h)), OUT_H
    fg = cv2.resize(frame, (fg_w, fg_h))
    y0, x0 = (OUT_H - fg_h) // 2, (OUT_W - fg_w) // 2
    canvas[y0:y0 + fg_h, x0:x0 + fg_w] = fg
    _brand(canvas, lift, score, grade)
    return canvas


def make_share_clip(src_video, out_dir, *, lift: str = "squat", score=None, grade=None) -> str:
    """Re-frame the annotated landscape clip into a branded 9:16 portrait clip for social.

    Returns the H.264 output path (transcoded for clean phone playback). Raises ``ValueError`` if
    the source can't be read.
    """
    src = Path(src_video)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise ValueError(f"Couldn't open video to share: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / f"{src.stem}_share_raw.mp4"
    final = out_dir / f"{src.stem}_share.mp4"

    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (OUT_W, OUT_H))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(_compose(frame, lift, score, grade))
        n += 1
    cap.release()
    writer.release()
    if n == 0:
        raise ValueError("No frames decoded from the source video.")
    return mediamod.transcode_h264(str(raw), str(final))
