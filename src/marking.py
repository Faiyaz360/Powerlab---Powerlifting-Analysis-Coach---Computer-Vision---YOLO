"""Plate-marking geometry for the web app's two-tap reticle.

Pure functions only (no Gradio, no OpenCV) so the "which tap does what" logic is unit-testable in
isolation. The app draws the reticle and wires the taps; the rule lives here.

The user marks the plate in two taps:
  1. first tap  -> drop the centre,
  2. second tap -> set the radius (its distance from the centre).
Touch-native (a tap works the same with a finger or a mouse) and needs no slider dragging.
"""
from __future__ import annotations

import math

MIN_PLATE_R = 10  # px — floor so a second tap landing on the centre can't zero the radius


def tap_to_seed(
    tap_state: int, x: int, y: int, cx: int, cy: int, r: int
) -> tuple[int, int, int, int]:
    """Advance the two-tap plate marker by one tap.

    ``tap_state`` says which tap this is: 0 = place the centre, 1 = set the radius. The returned
    next-state flips it so taps alternate (centre, edge, centre, edge, ...).

    Args:
        tap_state: 0 if this tap sets the centre, 1 if it sets the radius.
        x, y: where the user tapped (pixels on the displayed frame).
        cx, cy, r: the current circle, so we can keep the parts this tap doesn't change.

    Returns:
        (cx, cy, r, next_state) — the updated circle and the state for the next tap.
    """
    if tap_state == 0:
        return x, y, r, 1  # placed the centre; the next tap sizes it
    r = max(MIN_PLATE_R, int(round(math.hypot(x - cx, y - cy))))
    return cx, cy, r, 0  # sized it; the next tap restarts at the centre
