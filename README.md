# Powerlifting Form-Analysis Tool

Record a lift → a vision model tracks your body → get coaching feedback + metrics.
**Phase 1 (now):** local squat analysis. Later phases add a web app, GPU models, barbell speed.

See the full build plan at `C:\Users\user\.claude\plans\hey-so-basically-i-pure-teacup.md`.

## What it does (Phase 1)

Feed it a **side-on squat video** and it produces:
- an annotated video (skeleton, knee angle, rep counter, depth badge),
- `*_report.md` (per-rep table + coaching cues),
- `*_metrics.json` and a knee-angle plot.

## Setup (one time)

Dependencies are already installed in `.venv`. If you ever need to reinstall:

```powershell
& "C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

1. Drop a side-on squat video into `input/` (e.g. `input/squat.mp4`).
2. Run:

```powershell
.\.venv\Scripts\python.exe cli.py input/squat.mp4
```

3. Find results in `output/`.

## Recording tips (matters a lot)

- Camera **side-on**, on a tripod, at roughly hip height.
- Whole body + barbell in frame, decent lighting, plain background.
- **60 fps** if your phone allows it.

## Run the tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Project layout

```
cli.py            # local entry point
src/
  pipeline.py     # orchestrates a run
  pose.py         # MediaPipe pose (model interface — swappable later)
  angles.py       # joint-angle math (unit-tested)
  metrics.py      # depth, rep detection, tempo
  faults.py       # thresholds -> issue list
  coach.py        # issue list -> coaching cues
  render.py       # draws the annotated video
  report.py       # writes report.md + metrics.json + plot
tests/            # unit tests
input/  output/   # your videos / results (git-ignored)
```
