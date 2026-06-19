# Phase 3 — 3D pose (depth · symmetry · valgus) — Design / Spec

**Date:** 2026-06-18 · **Status:** locked (design approved) · **Owner:** beginner — keep code small, commented, phased.

## Goal
Add monocular **3D** pose on top of the existing 2D pipeline to measure what a side-on 2D camera
physically cannot: true **squat depth in 3D**, **left/right symmetry**, and **knee valgus (cave)** —
the last two only with a **front / ~45° clip**. 3D is **additive**, never a replacement for the 2D
angles that stay the source of truth.

## Non-goals (explicit — so we don't over-promise)
- **NOT** replacing 2D joint angles. A 2025 clinical benchmark shows monocular 3D carries ~8–9° mean
  joint-angle error (2–3× worse than 2D). Angles stay 2D.
- **NOT** spine ROUNDING. H36M-17 has only pelvis–spine–thorax (2 segments) → gross hinge, not
  curvature. True rounding needs a denser 3D model — out of scope here (see LESSONS: the 2D
  silhouette attempt was removed for the same honesty reason).
- **NOT** metric/absolute depth. The lifter output is root-relative, unit-scaled → use for ratios /
  symmetry / depth-ratio, and label everything 3D as an **estimate**.
- **NOT** live/real-time. Record-then-analyse, same as today.

## Honest value (what each 3D metric is worth)
| 3D-only metric | Real? | Catch |
|---|---|---|
| Knee valgus (cave) | yes — #1 win | needs a front/45° clip; invisible side-on |
| L/R symmetry | yes — with front/45° | side-on the far limb is occluded → low-confidence |
| Squat depth in 3D | marginal | 2D already measures depth well; 3D may not beat it |
| Spine rounding | no | 17-joint can't; needs a denser model (not this phase) |

**Key truth:** the payoff comes from the **camera angle**, not the model swap. 3D on side-on-only
clips adds little; 3D + a front/45° clip unlocks valgus + symmetry.

## Model decision (from research, 2025–2026)
- **Lifter: MotionAGFormer-L** (WACV 2024, **Apache-2.0**). Won the 2025 clinical joint-angle
  benchmark (RMSE 9.27°) over MotionBERT; H36M-17 in/out, `(T,17,3)`, variants XS/S/B/L
  (2.2M→19M params) for budgeting.
- **Fallback: MotionBERT** (Apache-2.0, same H36M data format) — hot-swappable behind the interface.
- **Avoid:** MixSTE (no license), OPFormer (non-commercial), OpenPose/Sapiens (non-commercial).
- **PerfectRep** = architecture reference only (form-analysis is a stub, repo dormant); mine its
  Fit3D preprocessing (243-frame, stride 81) only if we ever fine-tune.

## Architecture (all heavy compute on ZeroGPU — never the owner's PC)
```
upload clip + pick VIEW (side / front-or-45°)
  → pose.py     2D keypoints (YOLO26, ZeroGPU)                 [unchanged; COCO-17 already mapped]
  → lift3d.py   COCO-17→H36M-17 adapter → MotionAGFormer → (T,17,3) 3D   [new, ZeroGPU]
  → metrics3d   3D depth · L/R symmetry · valgus(front/45°) — all ESTIMATE-labeled
  → render/report  new 3D panel, view-aware honesty gate
```
- `lift3d.py` sits **behind the existing `pose.py` interface** (keep the rest model-agnostic).
- **COCO-17 → H36M-17 adapter** (the one real piece of glue, pure numpy): pelvis = mid-hips,
  thorax = mid-shoulders, spine = mid(pelvis,thorax), neck/head from nose+thorax; hips/knees/
  ankles/shoulders/elbows/wrists map 1:1. `pose.py` already maps COCO-17 → our slots, so we build
  H36M-17 from slots we already have.
- **View-aware honesty gate** (replaces the global off-axis reject): **side** clip → judge 2D
  sagittal metrics as now; **front/45°** clip → skip the 2D sagittal angles (invalid off-axis),
  show valgus/symmetry from 3D instead, labeled estimate. Gate becomes per-metric, not one reject.
- **Single view-tagged clip first** (reuse `history.view`, already defaults `'side'`); dual
  side+front upload is a later step.

## Deployment constraints (researched)
- **Plain PyTorch only.** MotionAGFormer/MotionBERT lift keypoints, not pixels — deps are just
  `torch`/`torchvision` + a checkpoint. **Do NOT add mmpose/mmcv** (they fail the ZeroGPU build —
  documented). ViTPose, if ever needed for 2D, has an mmpose-free HF `transformers` path.
- **Cheap stage.** The lifter on keypoints is <1–2 s for a 10 s clip; the 2D pass stays the cost.
- **ZeroGPU rules:** load the model **once at module level** (CUDA emulation makes `.to('cuda')` at
  import work); the GPU work must run inside the **`@spaces.GPU` handler** (see LESSONS — the
  decorator-theft bug); no `torch.compile`; pin torch to a ZeroGPU-supported version (2.8–2.11).
- **Reference Space:** `hysts/ViTPose-transformers` (ZeroGPU + Gradio + transformers) for the wiring
  pattern.

## Build order (incremental — each step ships; de-risk on Colab, not the PC)
1. **lift3d adapter + tests** — `lift3d` COCO-17→H36M-17, pure numpy, unit-tested with synthetic
   keypoints (like the ghost tests). *Runs on the owner's PC, safe (array math).*
2. **lift3d inference** — load MotionAGFormer, adapter→model→`(T,17,3)`. One-time sanity on a
   **free Colab GPU** on one saved clip (eyeball depth/limbs). *Not the owner's PC.*
3. **Deploy 3D behind a flag** — wire into `analyze` inside the existing `@spaces.GPU` window,
   model loaded once at module level. Default OFF. Validate on the live Space.
4. **View selector + valgus + symmetry** — add a side/front/45° dropdown; compute the 3D-only
   metrics; view-aware gate; label estimates. *The real payoff.*
5. **(later)** dual side+front upload; then honest 3D spine via a denser model.

## Data / persistence
- Reuse the existing `history.view` column (defaults `'side'`). New 3D metrics can be added as
  history columns later (the table auto-migrates) — not required for stages 1–4.

## Risks / open questions
- **COCO→H36M domain gap:** lifters were trained on H36M-style 2D detections, not COCO — small
  added error. MotionAGFormer's noise-robust pretraining absorbs most; part of why 3D is an estimate.
- **Far-limb occlusion side-on:** symmetry from a side clip estimates an occluded limb → keep it
  low-confidence until a front/45° clip exists.
- **Capture UX for front/45°:** start with one view-tagged clip; revisit dual-upload after stage 4.
- **ZeroGPU quota:** 3D adds GPU time; keep it opt-in / view-gated.

## Verification (per stage)
1. Adapter: unit tests — known COCO frame → expected H36M-17 (pelvis/thorax/spine midpoints, 1:1 joints).
2. Inference: Colab run on `squat-1` → 3D skeleton looks anatomically sane; depth tracks the descent.
3. Deploy: Space RUNNING; flag-on analysis returns without error; flag-off unchanged.
4. View metrics: a known front clip → valgus fires on a visibly caving rep, quiet on a clean one;
   2D sagittal metrics correctly suppressed for the front view.

## Build status — autonomous session, 2026-06-19
- **Stage 1 (adapter): DONE + tested.** `src/lift3d.to_h36m17` (our 33 slots → H36M-17, spine chain
  synthesised), `fill_missing`, `normalize_crop` — pure, 5 unit tests.
- **Stage 2 (lifter): BUILT + verified offline, NOT app-wired.** `src/lift3d.lift_to_3d` lifts a clip
  to `(T,17,3)` via **MotionBERT lite** (DSTformer) minimally vendored in `src/_motionbert/`
  (Apache-2.0; weights auto-download from HF `walterzhu/MotionBERT`, cached). Verified end-to-end on
  `squat-1` (1229 frames, CPU). Renders: `tools/lift3d_motionbert.py` (accurate) + `tools/lift3d_preview.py`
  (MediaPipe-world quick look).
  - **Model choice:** **MotionBERT lite** chosen over MotionAGFormer because its weights are HF-hosted
    (reliable unattended download) and it's the spec's named hot-swap fallback (same H36M interface).
    **MotionAGFormer remains the accuracy target — a drop-in `(T,17,3)` H36M swap.**
  - The YOLO backend emits no nose → the adapter's neck/head are untracked (conf 0, model-guessed); the
    previews don't draw them.
- **NOT done (needs a supervised + deploy session):** wire `lift_to_3d` into `analyze`/`pipeline`
  behind a flag (keep the GPU call INSIDE `@spaces.GPU` — see the decorator lesson); the view selector +
  view-aware honesty gate; the honest 3D metrics (depth / symmetry / valgus, all estimate-labelled);
  the MotionAGFormer accuracy swap; the on-Space ZeroGPU run.
