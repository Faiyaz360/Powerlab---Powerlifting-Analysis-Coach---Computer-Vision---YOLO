"""Ghost compare — overlay your CURRENT rep against your BEST-EVER rep at the key frame, so form
drift is visible at a glance.

HONEST by construction: it draws REAL tracked landmarks + the REAL bar path (the same data the live
skeleton uses), not a silhouette guess. The only transform is a hip-align + torso-scale so the two
reps sit on top of each other — that shows FORM difference, not where each lifter happened to stand.

Stored small: just the key frame's landmarks + a downsampled bar path per run (a few KB of JSON),
so it rides the existing history DB -> bucket snapshot with no new persistence.
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from . import pose as P

_BAR_SAMPLES = 60      # downsample the stored bar path to keep the blob small

# camera-side sagittal chain to draw for the compare (name pairs, resolved per side)
_SKEL_EDGES = [("NOSE", "SHOULDER"), ("SHOULDER", "ELBOW"), ("ELBOW", "WRIST"),
               ("SHOULDER", "HIP"), ("HIP", "KNEE"), ("KNEE", "ANKLE"), ("ANKLE", "FOOT")]
_CUR_COLOR = (60, 255, 255)     # current rep = bright yellow (BGR)
_GHOST_COLOR = (255, 210, 90)   # best-ever rep = cyan (BGR), drawn faint


def _hip_torso(xy):
    """Hip centre + torso length (shoulder->hip) from a (33,2) landmark frame. None if not visible."""
    pts = {i: xy[i] for i in (P.L_SHOULDER, P.R_SHOULDER, P.L_HIP, P.R_HIP)}
    if any(np.any(np.isnan(p)) for p in pts.values()):
        return None, None
    hip = (pts[P.L_HIP] + pts[P.R_HIP]) / 2.0
    sho = (pts[P.L_SHOULDER] + pts[P.R_SHOULDER]) / 2.0
    torso = float(np.hypot(*(sho - hip))) or 1.0
    return hip, torso


def build_blob(landmarks, key_frame, side, bar_xy) -> str | None:
    """Pack a rep into a tiny JSON string for later ghosting: the key frame's landmarks, the camera
    side, a hip anchor + torso length (for alignment), and a downsampled bar path. None if the key
    frame's pose is unusable."""
    if landmarks is None or key_frame is None or not (0 <= key_frame < len(landmarks)):
        return None
    xy = np.asarray(landmarks[key_frame, :, :2], float)
    hip, torso = _hip_torso(xy)
    if hip is None:
        return None
    bar = []
    if bar_xy is not None and len(bar_xy):
        b = np.asarray(bar_xy, float)
        keep = b[~np.any(np.isnan(b), axis=1)]
        if len(keep):
            idx = np.linspace(0, len(keep) - 1, min(_BAR_SAMPLES, len(keep))).astype(int)
            bar = keep[idx].round(1).tolist()
    blob = {
        "xy": np.where(np.isnan(xy), None, xy.round(1)).tolist(),
        "side": side,
        "hip": [round(float(hip[0]), 1), round(float(hip[1]), 1)],
        "torso": round(torso, 1),
        "bar": bar,
    }
    return json.dumps(blob, separators=(",", ":"))


def align(blob, cur_hip, cur_torso):
    """Transform a stored ghost so its hip + torso match the CURRENT rep — translate the ghost hip
    onto ``cur_hip`` and scale by ``cur_torso / ghost_torso``. Returns ``(xy, bar)`` in current-frame
    pixels (xy is an (N,2) array with NaN for missing joints; bar is an (M,2) array, possibly empty).
    Pure. ``blob`` is the dict (already json-loaded) or a JSON string."""
    if isinstance(blob, str):
        blob = json.loads(blob)
    g_hip = np.asarray(blob["hip"], float)
    scale = (float(cur_torso) / float(blob["torso"])) if blob.get("torso") else 1.0
    cur_hip = np.asarray(cur_hip, float)

    def _xf(arr):
        a = np.array(arr, dtype=object)
        out = np.full((len(arr), 2), np.nan, float)
        for i, p in enumerate(arr):
            if p is not None and p[0] is not None and p[1] is not None:
                out[i] = cur_hip + (np.array(p, float) - g_hip) * scale
        return out

    xy = _xf(blob["xy"])
    bar = _xf(blob["bar"]) if blob.get("bar") else np.empty((0, 2))
    return xy, bar


def _idx(name, side):
    if name == "NOSE":
        return P.NOSE
    return getattr(P, ("L_" if side == "left" else "R_") + name)


def _draw_skeleton(img, xy, side, color, thick):
    """Draw the camera-side sagittal chain from an (N,2) landmark array (NaN joints skipped)."""
    for a, b in _SKEL_EDGES:
        pa, pb = xy[_idx(a, side)], xy[_idx(b, side)]
        if not (np.any(np.isnan(pa)) or np.any(np.isnan(pb))):
            cv2.line(img, tuple(np.int32(pa)), tuple(np.int32(pb)), color, thick, cv2.LINE_AA)
    for name in {n for e in _SKEL_EDGES for n in e}:
        p = xy[_idx(name, side)]
        if not np.any(np.isnan(p)):
            cv2.circle(img, tuple(np.int32(p)), thick + 2, color, -1)


def _draw_bar(img, bar, color, thick, dashed=False):
    """Draw a bar path (solid, or dashed for the ghost) from an (M,2) array."""
    pts = bar[~np.any(np.isnan(bar), axis=1)] if len(bar) else bar
    for i in range(1, len(pts)):
        if dashed and i % 2:
            continue
        cv2.line(img, tuple(np.int32(pts[i - 1])), tuple(np.int32(pts[i])), color, thick, cv2.LINE_AA)


def draw_ghost_panel(cur_frame, cur_xy, side, ghost_blob, cur_bar=None):
    """The ghost-compare panel: your CURRENT rep (bright) over your BEST-EVER rep (faint cyan),
    hip-aligned + torso-scaled at the key frame. ``ghost_blob`` is the stored JSON (or dict). Returns
    a BGR image, or None if the current pose is unusable."""
    img = cur_frame.copy()
    cur_xy = np.asarray(cur_xy, float)[:, :2]
    cur_hip, cur_torso = _hip_torso(cur_xy)
    if cur_hip is None:
        return None
    thick = max(2, img.shape[1] // 360)
    blob = json.loads(ghost_blob) if isinstance(ghost_blob, str) else ghost_blob
    gxy, gbar = align(blob, cur_hip, cur_torso)
    _draw_bar(img, gbar, _GHOST_COLOR, thick, dashed=True)               # ghost path (dashed, under)
    _draw_skeleton(img, gxy, blob.get("side", side), _GHOST_COLOR, thick)  # ghost skeleton (faint)
    if cur_bar is not None:
        _draw_bar(img, np.asarray(cur_bar, float), _CUR_COLOR, thick)    # current path (solid)
    _draw_skeleton(img, cur_xy, side, _CUR_COLOR, thick + 1)             # current skeleton (bright, on top)
    s = max(0.5, img.shape[1] / 1100.0)
    cv2.putText(img, "GHOST  you = yellow  vs  best = cyan", (int(16 * s), int(34 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
    return img
