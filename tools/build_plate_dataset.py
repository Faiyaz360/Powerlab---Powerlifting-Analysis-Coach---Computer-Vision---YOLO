"""Auto-label a YOLO plate-detection dataset using the HSV detector.

Runs the (reliable, on a clean clip) HSV plate detector over video frames and writes each
confident detection as a YOLO bbox label. Train a detector on this so it generalises to
cluttered / same-colour-background clips where HSV fails.

Usage:
    python tools/build_plate_dataset.py input/deadlift-1.mov [more.mov ...]
"""
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.barbell import _detect_plate  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "dataset"
STRIDE = 3        # label every Nth frame
VAL_FRAC = 0.2


def main(videos):
    random.seed(0)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    n = 0
    for vid in videos:
        cap = cv2.VideoCapture(vid)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        min_r, max_r = int(h * 0.05), int(h * 0.25)
        min_area = np.pi * min_r * min_r * 0.5
        stem = Path(vid).stem
        f = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if f % STRIDE == 0:
                hit = _detect_plate(frame, np.array([np.nan, np.nan]), min_r, max_r, min_area)
                if hit is not None:
                    (x, y), r = hit
                    split = "val" if random.random() < VAL_FRAC else "train"
                    name = f"{stem}_{f:05d}"
                    cv2.imwrite(str(OUT / f"images/{split}/{name}.jpg"), frame)
                    label = f"0 {x / w:.6f} {y / h:.6f} {2 * r / w:.6f} {2 * r / h:.6f}\n"
                    (OUT / f"labels/{split}/{name}.txt").write_text(label)
                    n += 1
            f += 1
        cap.release()

    (OUT / "data.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\nnc: 1\nnames: [plate]\n"
    )
    print(f"wrote {n} labeled frames to {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:])
