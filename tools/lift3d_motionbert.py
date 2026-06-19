"""Accurate offline 3D preview — uses the packaged src.lift3d.lift_to_3d (MotionBERT, Apache-2.0).

Lifts our cached 2D keypoints to 3D and renders the squat bottom from side / 45 / front, so the 3D
the side camera can't see is visible. Weights auto-download (~62 MB lite, cached) on first run. Demo
only — not wired into the app. Self-contained on src/ (no vendor/ dependency).
"""
import glob
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from src import lift3d

OUT = "output/lift3d_motionbert.png"
lm = np.load(sorted(glob.glob(".posecache/squat-1_yolo_yolo26m*.npz"))[0])["landmarks"]
pose3d = lift3d.lift_to_3d(lm)                            # (T,17,3) via the packaged lifter
h = lift3d.to_h36m17(lm)
bottom = int(np.nanargmax(h[:, 0, 1]))                    # deepest = max pelvis image-y
pose = pose3d[bottom]
print(f"frames {len(pose3d)} | bottom {bottom} | z-range {pose[:, 2].min():+.3f}..{pose[:, 2].max():+.3f}")

_SKIP = {9, 10}                                           # nose/neck/head: untracked (YOLO has no nose)
bones = [(i, p) for i, p in enumerate(lift3d.H36M_PARENTS)
         if p >= 0 and i not in _SKIP and p not in _SKIP]


def draw(ax, p, azim, title):
    X, Y, Z = p[:, 0], p[:, 1], p[:, 2]
    for a, b in bones:
        ax.plot([X[a], X[b]], [Z[a], Z[b]], [-Y[a], -Y[b]], "-o", color="#22c55e", ms=3, lw=2.2)
    ax.set_title(title, color="#cfd6e6", fontsize=11)
    ax.view_init(elev=8, azim=azim); ax.set_box_aspect([1, 1, 1.4]); ax.set_axis_off()


fig = plt.figure(figsize=(12, 4.2)); fig.patch.set_facecolor("#0d0f14")
for i, (az, t) in enumerate([(0, "side (your camera)"), (45, "45 deg"), (90, "front (unseen)")]):
    ax = fig.add_subplot(1, 3, i + 1, projection="3d"); ax.set_facecolor("#0d0f14")
    draw(ax, pose, az, t)
fig.suptitle("Accurate 3D — squat-1 bottom (MotionBERT lift of your 2D keypoints)",
             color="#e6e9f0", fontsize=13)
fig.savefig(OUT, dpi=130, bbox_inches="tight", facecolor="#0d0f14")
print("saved", OUT)
