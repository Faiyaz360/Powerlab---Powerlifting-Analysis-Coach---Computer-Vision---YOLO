"""Instant OFFLINE 3D preview — MediaPipe world landmarks (real 3D, no download, CPU).

Runs MediaPipe on a clip, grabs the per-frame WORLD landmarks (metres, hip-centred 3D that our 2D
pipeline normally discards), finds the squat bottom, and renders that pose in 3D from side / 45 /
front so you can SEE the depth + left-right a side-on 2D camera can't. Approximate (MediaPipe 3D) —
the accurate MotionAGFormer lifter wires onto src/lift3d.to_h36m17 next. Demo only, not wired in.
"""
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

sys.path.insert(0, ".")
from src import pose as P

VIDEO = "input/squat-1.mov"
OUT = "output/lift3d_preview.png"
CONN = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
        (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32)]

model_path = P._ensure_model(2)
opts = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=str(model_path)),
    running_mode=vision.RunningMode.VIDEO, num_poses=1)

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
world, hipy = [], []
with vision.PoseLandmarker.create_from_options(opts) as lmk:
    idx, last = 0, -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        ts = max(last + 1, int(idx * 1000.0 / fps)); last = ts
        res = lmk.detect_for_video(img, ts)
        if res.pose_world_landmarks:
            wl = res.pose_world_landmarks[0]
            world.append(np.array([[p.x, p.y, p.z] for p in wl]))
            pl = res.pose_landmarks[0]
            hipy.append((pl[23].y + pl[24].y) / 2.0)
        else:
            world.append(np.full((33, 3), np.nan)); hipy.append(np.nan)
        idx += 1
cap.release()
world = np.array(world); hipy = np.array(hipy)
bottom = int(np.nanargmax(hipy))                     # deepest = largest image-y of the hips
print(f"frames {len(world)} | squat bottom at frame {bottom} ({bottom / fps:.1f}s)")

w = world[bottom]
# honest 3D readouts the side view can't give:
lk, rk = w[25], w[26]                                # knees
print(f"knee depth spread (z): L={lk[2]:+.3f} m  R={rk[2]:+.3f} m  | L-R knee x-gap {abs(lk[0]-rk[0])*100:.1f} cm")
print(f"L/R hip depth diff (symmetry hint): {abs(w[23][2]-w[24][2])*100:.1f} cm")


def draw(ax, w, azim, title):
    X, Y, Z = w[:, 0], w[:, 1], w[:, 2]
    for a, b in CONN:
        ax.plot([X[a], X[b]], [Z[a], Z[b]], [-Y[a], -Y[b]], "-o",
                color="#3aa0ff", ms=3, lw=2.2)
    ax.set_title(title, color="#cfd6e6", fontsize=11)
    ax.view_init(elev=8, azim=azim)
    ax.set_box_aspect([1, 1, 1.4]); ax.set_axis_off()


fig = plt.figure(figsize=(12, 4.2)); fig.patch.set_facecolor("#0d0f14")
for i, (az, t) in enumerate([(0, "side (your camera)"), (45, "45 deg"), (90, "front (unseen)")]):
    ax = fig.add_subplot(1, 3, i + 1, projection="3d"); ax.set_facecolor("#0d0f14")
    draw(ax, w, az, t)
fig.suptitle("3D preview — squat-1 bottom (MediaPipe world-3D, approximate)",
             color="#e6e9f0", fontsize=13)
fig.savefig(OUT, dpi=130, bbox_inches="tight", facecolor="#0d0f14")
print("saved", OUT)
