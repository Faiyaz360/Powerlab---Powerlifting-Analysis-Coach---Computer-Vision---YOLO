"""Persist user-marked plate circles as YOLO training labels — the data flywheel.

Seeding the tracker is required on every analysis, and each mark (plate centre + edge) is free
ground truth. We save the first frame as a JPG and a YOLO-format box label (class 0 = plate) so
``tools/train_plate.py`` can later fine-tune the detector. We only COLLECT here — no training
runs (no local GPU, per project rules).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np  # noqa: F401  (frames arrive as numpy arrays)


def circle_to_yolo(cx, cy, r, width, height):
    """Plate circle -> normalized YOLO box (xc, yc, w, h) in [0, 1] (pure — unit-tested)."""
    side = 2.0 * r
    return (cx / width, cy / height, side / width, side / height)


def save_label(frame_rgb, cx, cy, r, video_name, out_dir="data/plate_labels", stamp="") -> dict:
    """Write frame.jpg + YOLO label.txt for one marked plate. Returns the written paths.

    ``frame_rgb``: H x W x 3 RGB array (as shown in the UI). ``stamp``: unique suffix so repeated
    marks of the same clip don't overwrite each other.
    """
    out_dir = Path(out_dir)
    img_dir, lbl_dir = out_dir / "images", out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    h, w = frame_rgb.shape[:2]
    base = f"{video_name}_{stamp}" if stamp else video_name
    img_path = img_dir / f"{base}.jpg"
    lbl_path = lbl_dir / f"{base}.txt"

    cv2.imwrite(str(img_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    xc, yc, bw, bh = circle_to_yolo(cx, cy, r, w, h)
    lbl_path.write_text(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")
    return {"image": str(img_path), "label": str(lbl_path)}
