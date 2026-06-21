# Cloud (Colab) notebooks

Two notebooks, both on a **free Colab GPU** so nothing runs on your PC (no more GPU freezes):

- **`autolabel.ipynb`** — turn lifting clips into a YOLO plate dataset automatically (the data flywheel).
- **`rtmpose_eval.ipynb`** — run + score the pipeline with the Apache RTMPose backend vs YOLO.

---

## autolabel.ipynb — auto-label plates at scale

Feeds a folder of clips through an open-vocabulary detector (YOLO-World) that finds `"weight
plate"` in every frame, keeps the sane (round ⇒ square, sized) boxes, and writes a YOLO dataset
in the **exact layout `tools/train_plate.py` trains on**. Plus a QC report and a visual review grid.

1. Upload `colab/autolabel.ipynb` → **Runtime → GPU (T4)**.
2. **Get clips** (one of): the *gather* cell pulls CC-only clips right in Colab, **or** upload a
   zip of your `clips/` folder.
3. **(Optional) Cut long videos** — step 3c slices whole-session videos into per-set clips using
   body motion only, so you don't waste time scanning rest periods. Skip if clips are already short.
4. Run down: it labels → shows a review grid → zips `dataset.zip` to download.
4. Unzip into the repo's `dataset/`, then `python tools/train_plate.py` → `models/plate.pt`.

**Auto-labels plates only.** Pose joints already come from the model; eval ground truth (rep count,
depth/lockout pass) is human judgment — keep that manual in `eval/groundtruth/`.

### Automating the *gather* step
- One list of search terms lives in `tools/queries.txt`; run
  `python tools/gather_clips.py --queries-file tools/queries.txt` anytime — a download archive
  means re-runs only fetch **new** clips, and a `manifest.csv` catalogues them with their license.
- Hands-off weekly (Windows, run once in PowerShell):
  ```powershell
  schtasks /create /tn "lift-gather" /sc weekly /d SUN /st 09:00 ^
    /tr "cmd /c cd /d \"C:\path\to\Powerlifting-project\" && .\.venv\Scripts\python.exe tools\gather_clips.py --queries-file tools\queries.txt"
  ```

---

## rtmpose_eval.ipynb — RTMPose eval

Run the pipeline with the Apache **RTMPose** backend and score it against your eval set.

## How to use

1. **Zip the project folder** on your PC — exclude `.venv/`, `runs/`, `dataset/`, `.posecache/`,
   `output/` (keep `src/`, `eval/`, `cli.py`, `requirements.txt`, and `input/` with your videos).
2. Go to [colab.research.google.com](https://colab.research.google.com) → **File → Upload notebook**
   → upload `colab/rtmpose_eval.ipynb`.
3. **Runtime → Change runtime type → GPU (T4)**.
4. Run the cells top to bottom: it installs deps, you upload the zip, then it prints the accuracy
   scorecard for **RTMPose** and for the **YOLO** baseline so you can compare.

## Why cloud

- RTMPose runs through `rtmlib` + onnxruntime — clean on Colab's Linux GPU (the `onnxruntime-gpu`
  CUDA setup that's painful on Windows just works here).
- Training and heavy inference belong in the cloud; your RTX 2060 also drives your display, so
  sustained GPU load can freeze Windows.

## Note

The scorecard is only as meaningful as the eval set. Add 10-20+ varied labelled clips to
`eval/groundtruth/` (see `eval/README.md`) — including imperfect reps — before trusting the number.
