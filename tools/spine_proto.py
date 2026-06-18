"""PROTOTYPE (NOT wired into the app) — can we measure spinal ROUNDING from a body silhouette?

Segments the lifter (MediaPipe selfie segmenter) and, walking ALONG the shoulder->hip axis, marches
PERPENDICULAR into the back at each station to the silhouette edge -> the back-surface contour. A
neutral (flat) back gives a straight contour; ROUNDING makes it bow. We report the max bow off the
contour's own chord (works at any torso angle, incl. a horizontal deadlift). Saves debug frames so we
can EYEBALL whether the mask actually tracks the back through bar / plates / gym clutter.

Run: .\.venv\Scripts\python.exe tools/spine_proto.py input/deadlift-1.mov deadlift
Caches the slow pose pass to output/_pose_cache_<stem>.npz so re-runs only redo segmentation.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import metrics as metricsmod          # noqa: E402
from src import pose as P                       # landmark indices + estimate_pose  # noqa: E402

MODEL = "models/selfie_segmenter.tflite"


def _pose(video):
    cache = Path(f"output/_pose_cache_{Path(video).stem}.npz")
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return d["lm"], d["lean"], str(d["side"]), int(d["n"])
    pose = P.estimate_pose(video, backend="mediapipe")
    an = metricsmod.analyze(pose, sys.argv[2] if len(sys.argv) > 2 else "deadlift")
    lm, lean, side = pose.landmarks, np.array(an["series"]["lean"], float), an["series"]["side"]
    cache.parent.mkdir(exist_ok=True)
    np.savez(cache, lm=lm, lean=lean, side=np.array(side), n=pose.num_frames)
    return lm, lean, side, pose.num_frames


def _person_mask(seg, bgr):
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    res = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    m = res.confidence_masks[0].numpy_view()
    if m.shape != bgr.shape[:2]:
        m = cv2.resize(m, (bgr.shape[1], bgr.shape[0]))
    return m >= 0.5


def _back_contour(mask, sh, hip, knee):
    """Walk shoulder->hip; at each station march PERPENDICULAR into the back to the silhouette edge.
    Back direction comes from the KNEE: the belly/front faces the knees, so the back is the opposite
    side of the torso axis — robust at ANY torso angle (upright squat or horizontal deadlift)."""
    sh, hip, knee = np.array(sh, float), np.array(hip, float), np.array(knee, float)
    axis = hip - sh
    L = float(np.hypot(*axis)) or 1.0
    u = axis / L
    n = np.array([-u[1], u[0]])
    if np.dot(knee - sh, n) > 0:                        # knee on the +n side = front -> flip to back
        n = -n
    h, w = mask.shape
    pts = []
    for t in np.linspace(0.05, 0.95, 22):               # skip the very ends (neck/glute noise)
        base = sh + t * axis
        edge = None
        for d in range(int(1.4 * L)):
            xi, yi = int(round(base[0] + d * n[0])), int(round(base[1] + d * n[1]))
            if not (0 <= xi < w and 0 <= yi < h) or not mask[yi, xi]:
                break
            edge = (xi, yi)
        if edge:
            pts.append(edge)
    return pts


def _max_bow(pts):
    """Max perpendicular deviation of the back contour from its own first->last chord (signed)."""
    if len(pts) < 3:
        return 0.0, None, 1.0
    a, b = np.array(pts[0], float), np.array(pts[-1], float)
    ab = b - a
    L = float(np.hypot(*ab)) or 1.0
    nrm = np.array([ab[1], -ab[0]]) / L
    best, bp = 0.0, None
    for p in pts[1:-1]:
        dev = float(np.dot(np.array(p, float) - a, nrm))
        if abs(dev) > abs(best):
            best, bp = dev, p
    return best, bp, L


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else "input/deadlift-1.mov"
    lm, lean, side, nframes = _pose(video)

    f = int(np.nanargmax(lean))                         # deepest hinge = highest rounding risk
    sh = lm[f, P.L_SHOULDER if side == "left" else P.R_SHOULDER, :2]
    hip = lm[f, P.L_HIP if side == "left" else P.R_HIP, :2]
    knee = lm[f, P.L_KNEE if side == "left" else P.R_KNEE, :2]
    nose = lm[f, P.NOSE, :2]
    facing = "left" if (not np.isnan(nose[0]) and nose[0] < hip[0]) else "right"

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("could not read frame", f)
        return

    seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL),
        output_confidence_masks=True, output_category_mask=False))
    mask = _person_mask(seg, frame)
    pts = _back_contour(mask, sh, hip, knee)
    bow, bpt, L = _max_bow(pts)
    torso = float(np.hypot(hip[0] - sh[0], hip[1] - sh[1])) or 1.0
    pct = 100 * abs(bow) / torso

    dbg = frame.copy()
    cont, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(dbg, cont, -1, (90, 90, 90), 1)
    cv2.line(dbg, tuple(np.int32(sh)), tuple(np.int32(hip)), (0, 170, 255), 2)          # torso axis
    if len(pts) >= 2:
        cv2.line(dbg, pts[0], pts[-1], (180, 180, 180), 1)                              # back-edge chord
        for p in pts:
            cv2.circle(dbg, p, 3, (255, 100, 210), -1)                                  # back contour
    if bpt:
        cv2.circle(dbg, bpt, 8, (0, 0, 255), 2)                                         # deepest bow
    cv2.putText(dbg, f"back bow {abs(bow):.0f}px ({pct:.0f}% torso)  facing {facing}  pts {len(pts)}",
                (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    stem = Path(video).stem
    cv2.imwrite(f"output/_spine_proto_{stem}.png", dbg)
    mo = frame.copy()
    mo[mask] = (0.5 * mo[mask] + np.array([0, 90, 0])).astype(np.uint8)
    cv2.imwrite(f"output/_spine_mask_{stem}.png", mo)
    print(f"frame {f}/{nframes} side={side} facing={facing} lean={lean[f]:.0f} "
          f"bow={abs(bow):.1f}px ({pct:.1f}% torso) pts={len(pts)} -> output/_spine_proto_{stem}.png")


if __name__ == "__main__":
    main()
