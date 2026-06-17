"""Bar-path speed colormap (red = slow -> blue = avg -> green = fast), in BGR for OpenCV."""
import numpy as np

from src import render


def test_speed_color_slow_is_red():
    assert render._speed_color(0.0) == (0, 0, 255)


def test_speed_color_avg_is_blue():
    assert render._speed_color(0.5) == (255, 0, 0)


def test_speed_color_fast_is_green():
    assert render._speed_color(1.0) == (0, 255, 0)


def test_speed_color_clamps_out_of_range():
    assert render._speed_color(-1.0) == (0, 0, 255)
    assert render._speed_color(2.0) == (0, 255, 0)


def test_start_anchor_is_the_literal_lift_start():
    """The 'start' lines anchor to the literal moment the lift starts in rep 1: a squat's last frame
    near the standing top before it drops; a deadlift's floor valley. Not the walkout at the video
    start — so a squat's centre line isn't pinned to the rack. No reps -> first tracked frame."""
    bar_xy = np.full((50, 2), 100.0)                         # y=100 = standing height
    bar_xy[0:5, 0] = 300.0                                   # walkout: video-start X is off at the rack
    bar_xy[16:21, 1] = [130, 160, 190, 200, 200]            # descending into the bottom at frame 20
    reps = [{"bottom": 20, "top": 30}]
    valid = np.arange(50)
    f = render._start_anchor_frame(bar_xy, reps, "squat", valid)
    assert f == 15                                           # last frame still standing before the drop
    assert bar_xy[f, 0] == 100.0                             # post-walkout (rack X=300 skipped)
    assert render._start_anchor_frame(bar_xy, reps, "deadlift", valid) == 20   # deadlift -> floor/liftoff
    assert render._start_anchor_frame(bar_xy, [], "squat", valid) == 0         # no reps -> video start


def test_velocity_graph_draws_rep_fill_and_start_line():
    """The bottom velocity graph renders a green concentric-rep fill and a red dotted 'rep start'
    vertical without error, and actually marks the frame (green fill + red pixels appear)."""
    h, w, n = 400, 600, 60
    frame = np.zeros((h, w, 3), np.uint8)
    gx0, gx1 = int(w * 0.05), int(w * 0.95)
    gy1 = int(h * 0.97)
    gy0 = gy1 - int(h * 0.14)
    gmid = (gy0 + gy1) // 2
    vs = np.zeros(n)
    vs[20:40] = 1.0                                          # one positive hump = a concentric rep
    vmax = 1.0
    xs = gx0 + (gx1 - gx0) * np.arange(n) // max(1, n - 1)
    ys = np.clip((gmid - (vs / vmax) * ((gy1 - gy0) / 2) * 0.9).astype(int), gy0, gy1)
    pts = np.stack([xs, ys], axis=1).astype(np.int32)
    box = (gx0, gy0, gx1, gy1, gmid, vmax)

    render._draw_velocity_graph(frame, pts, box, n - 1, reps_idx=[(20, 39)])

    b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    assert ((r > 150) & (g < 80) & (b < 80)).any()           # red 'rep start' vertical present
    assert ((g > 50) & (g > b + 20) & (g > r + 20)).any()    # green concentric fill present
