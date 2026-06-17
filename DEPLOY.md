# Deploying PowerLab to Hugging Face Spaces

The app is packaged to run on a **free CPU Space**. This covers the one-time setup, the push,
and the GPU upgrade path.

## What's already prepped
- `requirements.txt` uses `opencv-python-headless` + `scipy` (server-safe, no display needed).
- `packages.txt` installs `ffmpeg` (needed to transcode HEVC phone clips for the browser preview).
- `README.md` carries the Space config in its YAML frontmatter (`sdk: gradio`, `app_file: app.py`).
- `app.py` auto-selects the **MediaPipe (CPU)** pose backend when Hugging Face's `SPACE_ID` env
  var is present, and **YOLO (GPU)** locally. Override either way with the `POSE_BACKEND` env var.

## One-time setup
1. Make a free account: https://huggingface.co/join
2. Create a **write** token: https://huggingface.co/settings/tokens
   (New token, role **Write**, copy it — this is your push password.)

## Create the Space
- Go to https://huggingface.co/new-space
- Name e.g. `form-lab`, SDK **Gradio**, Hardware **CPU basic (free)**, Public or Private, **Create**.

## Push the code
From the project folder:
```powershell
git remote add space https://huggingface.co/spaces/<your-username>/form-lab
git push space phase2-web-app:main
```
Username = your HF username, password = the **write token**. Hugging Face then builds it
(installs `requirements.txt` + `ffmpeg`, downloads the MediaPipe model on first run) and launches
`app.py`. Watch the **Logs** tab.

**No git?** On the Space: Files -> Add file -> upload `app.py`, `requirements.txt`, `packages.txt`,
and drag the `src/` folder.

## What to expect (free CPU tier)
- First analysis is slow (~30-90s): cold start + model download + CPU pose.
- The Space sleeps after ~48h idle and wakes on the next visit.
- History (`data/history.db`) resets on a rebuild (free Spaces have ephemeral disk).
- CPU MediaPipe is less sharp than your local GPU YOLO. Fine for testing; GPU is the upgrade.

## Updating later
Just push again: `git push space phase2-web-app:main`. Nothing is locked once uploaded.

## GPU upgrade path (sharper + faster, and required for the future 3D model)
1. Space -> Settings -> **Hardware** -> pick a GPU (e.g. Nvidia T4; set **sleep when idle** so you
   only pay during an actual analysis — pennies per lift).
2. Add `torch` + `ultralytics` to `requirements.txt` (kept out of the CPU build to keep it light).
3. Space -> Settings -> **Variables** -> set `POSE_BACKEND=yolo` (overrides the CPU default).

Same app, GPU pose.
