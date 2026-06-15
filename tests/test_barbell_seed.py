"""Ground-truth scale from a user-marked plate radius + the manual seed tracker."""
import cv2
import numpy as np

from src import barbell


def test_scale_from_seed_uses_plate_diameter():
    # r=225px -> diameter 450px maps to 0.450 m -> 0.001 m/px
    assert abs(barbell.scale_from_seed(225) - 0.001) < 1e-9


def test_scale_from_seed_none_on_bad_radius():
    assert barbell.scale_from_seed(0) is None
    assert barbell.scale_from_seed(None) is None


def test_track_from_seed_follows_a_moving_plate(tmp_path):
    """Synthetic clip: a blue disk with a white hub slides down. Seeded on frame 0, the manual
    template tracker should follow it (y rises, x stays put)."""
    w, h, n, r = 320, 240, 20, 30
    path = tmp_path / "syn.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
    first_y = 60
    for f in range(n):
        frame = np.full((h, w, 3), 128, np.uint8)                     # grey background
        cy = first_y + f * 6
        cv2.circle(frame, (160, cy), r, (255, 0, 0), -1)              # blue plate
        cv2.circle(frame, (160, cy), r // 3, (255, 255, 255), -1)     # white hub = template feature
        writer.write(frame)
    writer.release()

    centers, _ = barbell._track_from_seed(str(path), n, (160, first_y, r))

    assert centers[-1, 1] - centers[0, 1] > 50    # followed the disk downward
    assert abs(centers[-1, 0] - 160) < 15         # x stayed put
