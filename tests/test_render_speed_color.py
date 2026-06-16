"""Bar-path speed colormap (red = slow -> blue = avg -> green = fast), in BGR for OpenCV."""
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
