"""Two-tap plate-marking geometry (src/marking.py)."""
from src.marking import MIN_PLATE_R, tap_to_seed


def test_first_tap_sets_centre_and_advances():
    # Arrange: a fresh marker (state 0) with some default radius we expect kept untouched.
    # Act: tap at (200, 150).
    cx, cy, r, nxt = tap_to_seed(0, x=200, y=150, cx=0, cy=0, r=60)

    # Assert: centre jumps to the tap, radius unchanged, now waiting for the edge tap.
    assert (cx, cy) == (200, 150)
    assert r == 60
    assert nxt == 1


def test_second_tap_sets_radius_from_distance():
    # Arrange: centre already at (100, 100), state 1 (waiting for the edge).
    # Act: tap 30 px right + 40 px down -> 3-4-5 triangle -> radius 50.
    cx, cy, r, nxt = tap_to_seed(1, x=130, y=140, cx=100, cy=100, r=60)

    # Assert: centre kept, radius is the distance, back to waiting for a centre tap.
    assert (cx, cy) == (100, 100)
    assert r == 50
    assert nxt == 0


def test_radius_never_below_floor():
    # Arrange: the edge tap lands right on the centre (distance 0).
    # Act
    _, _, r, _ = tap_to_seed(1, x=100, y=100, cx=100, cy=100, r=60)

    # Assert: clamped to the minimum, never zero.
    assert r == MIN_PLATE_R
