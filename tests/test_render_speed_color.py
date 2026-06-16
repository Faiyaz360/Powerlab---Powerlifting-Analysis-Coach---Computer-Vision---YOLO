"""Bar-path speed colormap (blue = slow -> red = fast), in BGR for OpenCV."""
from src import render


def test_speed_color_slow_is_blue():
    assert render._speed_color(0.0) == (255, 0, 0)


def test_speed_color_mid_is_green():
    assert render._speed_color(0.5) == (0, 255, 0)


def test_speed_color_fast_is_red():
    assert render._speed_color(1.0) == (0, 0, 255)


def test_speed_color_clamps_out_of_range():
    assert render._speed_color(-1.0) == (255, 0, 0)
    assert render._speed_color(2.0) == (0, 0, 255)
