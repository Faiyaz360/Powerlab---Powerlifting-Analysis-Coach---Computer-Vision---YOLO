"""Train a YOLO11n plate detector on the auto-labelled dataset.

Strong HSV-hue augmentation (hsv_h) makes the detector colour-robust — it learns the plate's
round shape rather than the (blue) training colour, so it finds red/other plates too.

Usage: python tools/train_plate.py
"""
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(ROOT / "dataset" / "data.yaml"),
        epochs=40,
        imgsz=640,
        batch=16,
        hsv_h=0.5,    # full-range hue jitter -> colour invariance (the whole point)
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        degrees=5.0,
        device=0,
        project=str(ROOT / "runs"),
        name="plate",
        exist_ok=True,
        verbose=False,
    )
    print("done — best weights at runs/plate/weights/best.pt")


if __name__ == "__main__":
    main()
