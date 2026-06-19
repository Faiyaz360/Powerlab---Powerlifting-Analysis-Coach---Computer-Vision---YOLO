"""Vendored MotionBERT model (DSTformer) — Apache-2.0, Wenhao Zhu (github.com/Walter0807/MotionBERT).

Only the model DEFINITION is vendored so our 2D->3D lifter (``src.lift3d.lift_to_3d``) runs without
the full repo; the trained weights download from Hugging Face (walterzhu/MotionBERT) at first use and
cache, the same auto-download pattern as the pose models. ``drop.py`` is timm's DropPath (Apache-2.0,
Ross Wightman). No code here is ours — see the upstream repo for the licence.
"""
