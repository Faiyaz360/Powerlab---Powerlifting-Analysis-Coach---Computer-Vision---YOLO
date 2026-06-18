"""Spine Stage 2a — the back's SILHOUETTE curve.

Stage 1 (`render._draw_back_line`) draws a straight shoulder->hip line: it gives the back LEAN
angle but, being two points, can't show the back's actual shape. Stage 2a segments the lifter and
traces the BACK outline, so the overlay follows the real curve of the back.

HONEST SCOPE: this is a VISUAL. It does NOT yet JUDGE rounding (that's Stage 2b — the bow->lumbar
-rounding mapping still has to be validated against neck/traps/shirt confounds). Experimental,
opt-in. Segmentation = MediaPipe selfie segmenter (a ~250 KB model); kept behind this module so the
rest of the pipeline stays model-agnostic, and degrades gracefully (``available()`` is False, the
caller falls back to the straight Stage-1 line) when the model or mediapipe is missing.

Method (de-risked in tools/spine_proto.py): walk the shoulder->hip axis; at each station march
PERPENDICULAR into the back to the silhouette edge -> the back contour. Back side = OPPOSITE the
knee, so it's robust at any torso angle (upright squat or horizontal deadlift). The barbell plate
FUSES into the person mask, so the tracked plate disc is subtracted first. See LESSONS.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "selfie_segmenter.tflite"
_STATIONS = 22          # sample points along the torso axis
_T0, _T1 = 0.08, 0.95   # skip the very ends (neck / glute noise)


def back_contour(mask, sh, hip, knee):
    """Back-surface contour points (list of ``(x, y)``) or None.

    Walk shoulder->hip; at each station march PERPENDICULAR into the back to the silhouette edge.
    Back direction comes from the KNEE — the belly/front faces the knees, so the back is the
    opposite side of the torso axis. Pure (takes a boolean mask) — unit-tested on synthetic masks.
    """
    sh = np.asarray(sh, float)
    hip = np.asarray(hip, float)
    knee = np.asarray(knee, float)
    axis = hip - sh
    length = float(np.hypot(*axis))
    if length < 1.0:                                   # shoulder == hip -> no axis
        return None
    u = axis / length
    n = np.array([-u[1], u[0]])                        # perpendicular to the torso axis
    if np.dot(knee - sh, n) > 0:                       # knee on the +n side = front -> flip to back
        n = -n
    h, w = mask.shape
    pts = []
    for t in np.linspace(_T0, _T1, _STATIONS):
        base = sh + t * axis
        edge = None
        for d in range(int(1.4 * length)):
            xi = int(round(base[0] + d * n[0]))
            yi = int(round(base[1] + d * n[1]))
            if not (0 <= xi < w and 0 <= yi < h) or not mask[yi, xi]:
                break
            edge = (xi, yi)                            # last point still inside = the back surface
        if edge is not None:
            pts.append(edge)
    return pts if len(pts) >= 3 else None


def subtract_disc(mask, disc):
    """Clear a circle (the barbell plate, which the segmenter fuses into the body) from the mask.
    ``disc`` = ``(cx, cy, r)``. Returns a NEW mask (the original is left untouched)."""
    cx, cy, r = disc
    if not r or r <= 0:
        return mask
    out = mask.astype(np.uint8)
    cv2.circle(out, (int(round(cx)), int(round(cy))), int(round(r)), 0, -1)
    return out.astype(bool)


def smooth(pts, k=3):
    """Light moving-average along the contour — tames the staircase wobble of a raw silhouette
    edge for a clean draw. Window shrinks at the ends. Pure; returns a list of ``(x, y)`` ints."""
    if pts is None or len(pts) < 3:
        return pts
    a = np.asarray(pts, float)
    n = len(a)
    pad = k // 2
    out = []
    for i in range(n):
        lo, hi = max(0, i - pad), min(n, i + pad + 1)
        out.append(tuple(a[lo:hi].mean(axis=0).astype(int)))
    return out


def curvature(pts):
    """Per-point local turning angle (degrees) along the contour — how sharply the back outline
    changes direction at each point. 0 = locally straight, larger = a sharper bend. The endpoints
    have no defined turn, so they're 0. Pure; returns a list aligned with ``pts``.

    It's an ANGLE (scale-free), so the values are stable regardless of how big the lifter is on
    screen. HONEST SCOPE: this measures the silhouette's local bend — it is NOT a rounding verdict.
    """
    n = len(pts) if pts is not None else 0
    if n < 3:
        return [0.0] * n
    a = np.asarray(pts, float)
    out = [0.0]
    for i in range(1, n - 1):
        v1, v2 = a[i] - a[i - 1], a[i + 1] - a[i]
        m1, m2 = float(np.hypot(*v1)), float(np.hypot(*v2))
        if m1 < 1e-6 or m2 < 1e-6:
            out.append(0.0)
            continue
        cos = max(-1.0, min(1.0, float(np.dot(v1, v2) / (m1 * m2))))
        out.append(float(np.degrees(np.arccos(cos))))   # angle between successive segments
    out.append(0.0)
    return out


class BackCurve:
    """Lazy MediaPipe selfie-segmenter that extracts the back silhouette contour.

    ``available()`` is False — and the caller falls back to the straight Stage-1 line — when the
    model file or mediapipe is missing, so a segmentation problem never crashes the pipeline. The
    segmenter is created once and reused across frames.
    """

    def __init__(self, model_path=_MODEL_PATH):
        self._model_path = Path(model_path)
        self._seg = None
        self._mp = None
        self._tried = False

    def available(self) -> bool:
        return self._segmenter() is not None

    def _segmenter(self):
        if not self._tried:
            self._tried = True
            if self._model_path.exists():
                try:
                    import mediapipe as mp
                    from mediapipe.tasks import python as mp_python
                    from mediapipe.tasks.python import vision

                    self._mp = mp
                    self._seg = vision.ImageSegmenter.create_from_options(
                        vision.ImageSegmenterOptions(
                            base_options=mp_python.BaseOptions(
                                model_asset_path=str(self._model_path)),
                            output_confidence_masks=True, output_category_mask=False))
                except Exception:
                    self._seg = None
        return self._seg

    def person_mask(self, bgr):
        """Boolean person mask for a BGR frame, or None if the segmenter is unavailable/empty."""
        seg = self._segmenter()
        if seg is None:
            return None
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        res = seg.segment(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        if not res.confidence_masks:
            return None
        m = res.confidence_masks[0].numpy_view()
        if m.shape != bgr.shape[:2]:
            m = cv2.resize(m, (bgr.shape[1], bgr.shape[0]))
        return m >= 0.5

    def curve(self, bgr, sh, hip, knee, plate=None):
        """Back-silhouette contour for a frame, or None. ``plate`` = ``(cx, cy, r)`` of the barbell
        disc to subtract (it fuses into the person silhouette)."""
        mask = self.person_mask(bgr)
        if mask is None:
            return None
        if plate is not None:
            mask = subtract_disc(mask, plate)
        return back_contour(mask, sh, hip, knee)
