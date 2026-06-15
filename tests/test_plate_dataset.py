"""Plate-mark -> YOLO training label."""
import numpy as np

from src import plate_dataset


def test_circle_to_yolo_centre_and_size():
    # centre (50,100), r=25 in a 200x400 frame -> box side = 50px
    xc, yc, w, h = plate_dataset.circle_to_yolo(50, 100, 25, 200, 400)
    assert (xc, yc) == (0.25, 0.25)
    assert (w, h) == (0.25, 0.125)


def test_save_label_writes_image_and_label(tmp_path):
    frame = np.zeros((400, 200, 3), dtype=np.uint8)
    out = plate_dataset.save_label(frame, 50, 100, 25, "clip", out_dir=str(tmp_path), stamp="x")

    img_p = tmp_path / "images" / "clip_x.jpg"
    lbl_p = tmp_path / "labels" / "clip_x.txt"
    assert img_p.exists() and lbl_p.exists()
    assert out["image"] == str(img_p)

    parts = lbl_p.read_text().strip().split()
    assert parts[0] == "0"                       # class 0 = plate
    assert abs(float(parts[1]) - 0.25) < 1e-6    # xc
    assert abs(float(parts[3]) - 0.25) < 1e-6    # w = 2r/W = 50/200
