"""3D pose — Stage 1: map our 33-slot 2D landmarks to the H36M-17 skeleton a 2D->3D lifter wants.

The lifter (MotionAGFormer / MotionBERT, added next) consumes the 17-joint Human3.6M skeleton, but
our pipeline speaks the 33 MediaPipe-style slots (``pose.py``). This adapter bridges them by
SYNTHESISING the spine chain H36M needs (pelvis, thorax, spine, neck, head) as midpoints /
extrapolations of the joints we already track — the standard COCO/Blaze -> H36M convention. Pure +
deterministic, so it unit-tests with no model and runs anywhere (no GPU, no download).
"""
from __future__ import annotations

import numpy as np

from . import pose as P

# H36M-17 joint order (the MotionBERT / MotionAGFormer convention).
H36M_NAMES = ["Hip", "RHip", "RKnee", "RFoot", "LHip", "LKnee", "LFoot", "Spine", "Thorax",
              "Neck", "Head", "LShoulder", "LElbow", "LWrist", "RShoulder", "RElbow", "RWrist"]

# H36M bone connections (parent indices) for drawing / sanity checks.
H36M_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]


def to_h36m17(landmarks: np.ndarray) -> np.ndarray:
    """``(N, 33, >=2)`` our-slot landmarks -> ``(N, 17, 2)`` H36M-17 (x, y in pixels).

    Pelvis = mid-hips, thorax = mid-shoulders, spine = their midpoint, neck ~ nose, head extrapolated
    just above the nose. NaN propagates where a source joint is missing (caller fills before lifting).
    """
    lm = np.asarray(landmarks, dtype=float)[:, :, :2]

    def j(idx):
        return lm[:, idx, :]

    pelvis = (j(P.L_HIP) + j(P.R_HIP)) / 2.0
    thorax = (j(P.L_SHOULDER) + j(P.R_SHOULDER)) / 2.0
    spine = (pelvis + thorax) / 2.0
    nose = j(P.NOSE)
    head = nose + (nose - thorax) * 0.3        # a head-top a little above the nose

    return np.stack([
        pelvis, j(P.R_HIP), j(P.R_KNEE), j(P.R_ANKLE),
        j(P.L_HIP), j(P.L_KNEE), j(P.L_ANKLE),
        spine, thorax, nose, head,
        j(P.L_SHOULDER), j(P.L_ELBOW), j(P.L_WRIST),
        j(P.R_SHOULDER), j(P.R_ELBOW), j(P.R_WRIST),
    ], axis=1)


# --- Stage 2: lift the 2D keypoints to 3D with MotionBERT (Apache-2.0; weights auto-download) ------
# Monocular 3D is an ESTIMATE (reported joint angles stay 2D — see docs/phase3-3d-pose-design.md);
# this adds the depth + left/right a side view can't. Weights (~62 MB lite) download from HF on first
# call and cache. torch is imported lazily so the pure adapter above still runs with no torch present.
_H36M_LEFT, _H36M_RIGHT = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]
_MB_REPO = "walterzhu/MotionBERT"
_MB_FILE = "checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin"
_CLIP_LEN = 243
_model = None   # DSTformer, built once and cached


def fill_missing(h2d: np.ndarray):
    """Linear-interpolate each joint/axis over time (constant-fill a joint seen once; 0 if never seen).

    ``h2d``: ``(T, 17, 2)`` with NaN where a joint was untracked. Returns ``(filled, present)`` where
    ``filled`` has no NaN and ``present`` is a ``(T, 17)`` bool mask of the genuinely tracked joints.
    """
    h = np.asarray(h2d, dtype=float).copy()
    present = ~np.isnan(h[..., 0])
    t = np.arange(h.shape[0])
    for jt in range(h.shape[1]):
        for c in range(h.shape[2]):
            col = h[:, jt, c]
            ok = ~np.isnan(col)
            if ok.sum() >= 2:
                col[~ok] = np.interp(t[~ok], t[ok], col[ok])
            elif ok.sum() == 1:
                col[:] = col[ok][0]
    return np.nan_to_num(h, nan=0.0), present


def normalize_crop(motion: np.ndarray) -> np.ndarray:
    """MotionBERT's [-1, 1] normalisation (``crop_scale`` with scale_range=[1,1]). ``motion``:
    ``(T, 17, 3)`` = (x, y, conf). The bbox uses only confident joints; aspect is preserved. Pure."""
    res = np.asarray(motion, dtype=float).copy()
    valid = res[res[..., 2] != 0][:, :2]
    if len(valid) < 4:
        return np.zeros_like(res)
    xmin, xmax = valid[:, 0].min(), valid[:, 0].max()
    ymin, ymax = valid[:, 1].min(), valid[:, 1].max()
    scale = max(xmax - xmin, ymax - ymin) or 1.0
    xs, ys = (xmin + xmax - scale) / 2.0, (ymin + ymax - scale) / 2.0
    res[..., :2] = ((res[..., :2] - [xs, ys]) / scale - 0.5) * 2.0
    return np.clip(res, -1.0, 1.0)


def _flip3(x):
    """Horizontal-flip TTA on a torch ``(N, T, 17, C)`` clip: negate X, swap left/right joints."""
    f = x.clone()
    f[..., 0] *= -1
    f[:, :, _H36M_LEFT + _H36M_RIGHT] = f[:, :, _H36M_RIGHT + _H36M_LEFT]
    return f


def _load_model(device: str = "cpu"):
    """Build DSTformer once and cache it; weights download from HF (cached) on first call."""
    global _model
    if _model is None:
        import torch
        from huggingface_hub import hf_hub_download
        from ._motionbert.DSTformer import DSTformer
        net = DSTformer(dim_in=3, dim_out=3, dim_feat=256, dim_rep=512, depth=5, num_heads=8,
                        mlp_ratio=4, num_joints=17, maxlen=_CLIP_LEN, att_fuse=True)
        state = torch.load(hf_hub_download(_MB_REPO, _MB_FILE), map_location="cpu")["model_pos"]
        net.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=True)
        _model = net.to(device).eval()
    return _model


def lift_to_3d(landmarks: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Lift our ``(T, 33, 3)`` 2D landmarks to ``(T, 17, 3)`` root-relative 3D via MotionBERT.

    The whole clip runs in 243-frame windows (the model's length) with horizontal-flip TTA. ESTIMATE
    — monocular depth is approximate and must be labelled as such; reported joint angles stay 2D.
    Weights auto-download (~62 MB, cached) on the first call. Returns normalised 3D (not metres).
    """
    import torch
    filled, present = fill_missing(to_h36m17(landmarks))
    motion = np.concatenate([filled, present[..., None].astype(float)], axis=-1)   # (T,17,3)
    n_frames = motion.shape[0]
    model = _load_model(device)
    out = np.zeros((n_frames, 17, 3), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n_frames, _CLIP_LEN):
            win = motion[s:s + _CLIP_LEN]
            k = len(win)
            if k < _CLIP_LEN:
                win = np.pad(win, ((0, _CLIP_LEN - k), (0, 0), (0, 0)), mode="edge")
            x = torch.from_numpy(normalize_crop(win).astype(np.float32))[None].to(device)
            pred = ((model(x) + _flip3(model(_flip3(x)))) / 2.0)[0].cpu().numpy()
            out[s:s + k] = pred[:k]
    return out
