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


def test_detect_plate_seed_hough_finds_a_matte_plate():
    """A black (unsaturated) plate gives the HSV colour detector nothing, so the seed finder must
    fall back to Hough (shape) and still locate it — the case the user hits with non-vivid plates."""
    h, w = 240, 320
    frame = np.full((h, w, 3), 200, np.uint8)              # light-grey background
    cv2.circle(frame, (160, 120), 40, (0, 0, 0), -1)       # matte black plate, low saturation
    mr = int(h * 0.05)
    # the colour detector alone sees nothing on a dull plate...
    assert barbell._detect_plate(frame, np.array([np.nan, np.nan]), mr, int(h * 0.25),
                                 np.pi * mr * mr * 0.5) is None
    # ...but the seed finder falls back to Hough and locates it within the plate-size band
    hit = barbell.detect_plate_seed(frame, h)
    assert hit is not None
    cx, cy, r = hit
    assert abs(cx - 160) < 20 and abs(cy - 120) < 20
    assert mr <= r <= int(h * 0.25)


def test_detect_bar_reps_liftoff_is_after_floor_rest():
    """The bar rests on the floor, then lifts: the rep's `liftoff` (bar off the ground) is near the
    rise, NOT the early valley that sits in the floor rest — the red 'rep start' marker source."""
    n, fps, scale = 60, 30.0, 0.002
    y = np.full(n, 200.0)                                  # bar on the floor (high y)
    y[40:46] = [180, 160, 140, 120, 110, 100]             # pull up to the top at frame 45
    y[46:52] = [110, 130, 160, 190, 200, 200]             # lower back down
    bar_xy = np.column_stack([np.full(n, 50.0), y])
    reps = barbell.detect_bar_reps(bar_xy, fps, scale, min_rom_m=0.05)
    assert reps
    r = reps[0]
    assert r["liftoff"] > r["bottom"]                     # liftoff after the floor-rest valley
    assert r["liftoff"] >= 38                             # near where the pull actually begins (~40)


def test_velocity_per_rep_keeps_slow_reps_and_stays_aligned():
    """Regression (the 'rep 2 vanished' bug): a SLOW (heavy/grinder) rep is kept with its velocity,
    and the output is 1:1 with reps — a bad rep is None IN PLACE, never removed — so it can't shift
    the numbering of the reps after it."""
    fps, scale, n = 30.0, 0.002, 80
    y = np.full(n, 100.0)
    y[14:20] = 140.0                              # bar resting on the floor
    y[20:41] = np.linspace(140.0, 100.0, 21)      # a slow pull to lockout (~0.12 m/s, under old 0.3)
    bar_xy = np.column_stack([np.full(n, 50.0), y])
    reps = [{"bottom": 2, "top": 2},              # degenerate rep -> None, but holds its slot
            {"bottom": 17, "top": 40}]            # the slow real rep
    out = barbell.velocity_per_rep(bar_xy, reps, fps, scale, "deadlift")
    assert len(out) == len(reps)                  # nothing dropped -> later reps can't be renumbered
    assert out[0] is None
    assert out[1] is not None and out[1]["mean_velocity_ms"] > 0   # slow rep kept with its velocity


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
