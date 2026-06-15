# Data sourcing — playbook

This is a **commercial** product, so data must be **legally defensible**. Rule of thumb:
licensed datasets + Creative-Commons clips (with an audit trail) + consented user uploads — yes.
Bulk-scraping standard YouTube / Instagram / TikTok to train — **no** (ToS + copyright risk; US
Copyright Office May-2025 report says commercial training on scraped video isn't automatically
fair use, and there are active lawsuits naming yt-dlp).

## Green-light datasets (start here)

| Dataset | What | License | Commercial? | Link |
|---|---|---|---|---|
| **Kinetics-700-2020** | Real clips: `deadlifting`, `squat`, `bench pressing`, `clean and jerk`, `snatch` | CC BY 4.0 (attribute Google) | ✅ | https://github.com/cvdfoundation/kinetics-dataset |
| **InfiniteRep** | 1,000 synthetic exercise videos (squat, press) + 3D keypoints, rep counts | CC BY 4.0 | ✅ | https://github.com/toinfinityai/InfiniteRep |
| **ASPset-510** | 510 sport clips, clean 3D keypoints (pretraining a pose backbone) | CC0 (public domain) | ✅ | https://github.com/anibali/aspset-510 |
| **Roboflow barbell sets** | ~1.5–2k labelled barbell/plate images (merge HackerFit + Computer-Vision + kakann) | CC BY 4.0 (verify each) | ✅ | https://universe.roboflow.com/ (search "barbell") |
| **Fit3D** | Barbell exercises with VICON 3D ground truth (the most relevant) | Signed license | ⚠️ negotiate | https://fit3d.imar.ro/ |
| **OpenPowerlifting** | Meet results (no video) — useful for context/norms | Public domain | ✅ | https://www.openpowerlifting.org/ |

**Avoid for commercial:** SportsPose / AthletePose3D (academic-only), and any standard-license
YouTube/IG/TikTok scraping.

## Recommended order
1. **Kinetics lifting classes** (CC BY) — real squat/deadlift/bench footage, free, immediate.
2. **InfiniteRep** + **ASPset-510** — pose/keypoint pretraining data.
3. **Roboflow barbell sets** — bootstrap the plate detector (needs a free Roboflow API key).
4. **CC-only YouTube clips** via `tools/gather_clips.py` (below) — supplements, with license JSON kept.
5. **Email Fit3D** for a commercial license (slow — start early).
6. **Consented user uploads** (the flywheel) — your real moat once the app is live.

## Angle rule (important)
- **Detector training data → ALL angles welcome** (diagonal/side/front). A plate is round-ish from
  any view, so varied angles make the detector *more* robust.
- **Eval ground truth → dead side-on ONLY.** Form metrics (depth, joint angles, lockout, bar speed)
  are 2D — an off-axis camera foreshortens them, so a diagonal clip's depth/lockout label is invalid.
  Arbitrary angles only become valid for metrics with **3D pose** (ViTPose→MotionBERT, Phase 3).

## Tools in this repo (the data flywheel)
- `tools/fetch_roboflow.py` — pulls a **labelled** barbell/plate set from Roboflow Universe (YOLO
  format) → trains the plate detector directly, no video. Needs a free API key + `pip install roboflow`.
  **The fastest legal way to a working detector.**
- `tools/fetch_kinetics.py` — downloads the powerlifting classes from **Kinetics-700** (squat,
  deadlifting, bench pressing, clean and jerk, snatch weight lifting), each trimmed to its window.
  Varied-angle real lifts for pose/rep variety + segmentation testing. Research use (videos keep
  their copyright). Verified working: pulls real clips via `yt-dlp` (no JS runtime needed here).
- `tools/queries.txt` — the curated CC search terms (edit once).
- `tools/gather_clips.py` — downloads **only Creative-Commons** YouTube clips for those terms,
  **dedupes across runs** (a download archive), filters out long videos, and writes `manifest.csv`
  + each clip's license `.info.json` as an audit trail. Re-runnable/schedulable. Needs `yt-dlp`.
  `python tools/gather_clips.py --queries-file tools/queries.txt`
- `tools/segment_clips.py` — **cuts long videos into per-set clips** so we don't scan 20 min of
  resting/walking. Runs a fast pose model, finds the bursts where hips bounce (= a set), ffmpeg-snips
  each into its own clip. Body-motion only (no plate detector needed), so it runs before labeling.
  `python tools/segment_clips.py --in-dir raw --out clips`  (also a cell in `autolabel.ipynb`)
- `colab/autolabel.ipynb` — **auto-labels plates** on a free Colab GPU (YOLO-World finds plate
  boxes → sane ones written as a YOLO dataset in `tools/train_plate.py`'s format) + QC report +
  review grid. Then `tools/train_plate.py` fine-tunes the detector.
- **Stays human:** eval ground truth (rep count, depth/lockout pass) — the thing we measure —
  lives in `eval/groundtruth/` by hand (see `eval/README.md`). The auto-labeller can't judge it.

Loop: **gather (CC)** → **segment (cut to per-set clips)** → **auto-label plates (Colab)** → **review grid** → **train** → **measure on eval** → repeat.

## Legal audit trail
For every gathered clip, keep its `.info.json` (proves the CC license at download time). For user
uploads, store the consent-version + a ToS clause granting training rights (lawyer-reviewed).
