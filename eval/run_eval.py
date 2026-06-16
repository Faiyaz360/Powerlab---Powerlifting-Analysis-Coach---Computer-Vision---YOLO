"""Evaluation harness — measure pipeline accuracy against ground-truth labels.

Why: "accurate" must be a number, not a vibe. Add one JSON per clip under eval/groundtruth/
(see eval/README.md), then run:

    .\.venv\Scripts\python.exe eval\run_eval.py                 # one backend (default yolo)
    .\.venv\Scripts\python.exe eval\run_eval.py --pose rtmpose  # a specific backend
    .\.venv\Scripts\python.exe eval\run_eval.py --compare       # side-by-side table of all three

Reports rep-count accuracy and per-rep legal-pass accuracy (squat depth / deadlift lockout).
Runs the normal pipeline (uses the pose cache, so it's fast and inference-only).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.pipeline import analyze  # noqa: E402

GT_DIR = ROOT / "eval" / "groundtruth"
LEGAL_KEY = {"squat": "depth_pass", "deadlift": "lockout_pass"}  # the IPF red-light criterion
BACKENDS = ("mediapipe", "yolo", "rtmpose")


def _pred_legal(result, lift):
    return [bool(r[LEGAL_KEY[lift]]) for r in result["analysis"]["rep_metrics"]]


def score_backend(backend, files) -> dict:
    """Run one backend over every labelled clip and tally rep-count + depth/lockout accuracy."""
    rep_ok = 0
    legal_total = legal_correct = 0
    per_clip = []
    for f in files:
        gt = json.loads(f.read_text(encoding="utf-8"))
        lift = gt["lift"]
        res = analyze(str(ROOT / gt["video"]), lift=lift, backend=backend)
        pred, true = res["rep_count"], gt["true_rep_count"]
        count_ok = pred == true
        rep_ok += int(count_ok)
        clip = {"name": f.stem, "pred": pred, "true": true, "count_ok": count_ok,
                "legal_correct": None, "legal_n": None}

        gt_pass = gt.get("per_rep_legal_pass")
        if count_ok and gt_pass is not None:
            pred_pass = _pred_legal(res, lift)
            if len(pred_pass) == len(gt_pass):
                correct = sum(int(a == b) for a, b in zip(pred_pass, gt_pass))
                legal_total += len(gt_pass)
                legal_correct += correct
                clip["legal_correct"], clip["legal_n"] = correct, len(gt_pass)
        per_clip.append(clip)

    return {"backend": backend, "n": len(files), "rep_ok": rep_ok,
            "legal_correct": legal_correct, "legal_total": legal_total, "per_clip": per_clip}


def _rep_str(s):
    return f"{s['rep_ok']}/{s['n']} ({100 * s['rep_ok'] / s['n']:.0f}%)" if s["n"] else "-"


def _legal_str(s):
    if not s["legal_total"]:
        return "-"
    return f"{s['legal_correct']}/{s['legal_total']} ({100 * s['legal_correct'] / s['legal_total']:.0f}%)"


def print_single(s):
    print(f"Pose backend: {s['backend']}\n")
    for c in s["per_clip"]:
        line = f"{c['name']}: reps pred={c['pred']} true={c['true']} {'OK' if c['count_ok'] else 'MISS'}"
        if c["legal_n"]:
            line += f" | depth/lockout {c['legal_correct']}/{c['legal_n']}"
        print(line)
    print("\n=== Scorecard ===")
    print(f"Rep-count accuracy:      {_rep_str(s)}")
    print(f"Depth/lockout accuracy:  {_legal_str(s)}"
          if s["legal_total"] else "Depth/lockout accuracy:  (no aligned reps)")


def print_compare(scores):
    print("=== Backend comparison ===\n")
    hdr = f"{'backend':<12}{'rep acc':>16}{'depth/lockout':>18}"
    print(hdr)
    print("-" * len(hdr))
    for s in scores:
        if s.get("error"):
            print(f"{s['backend']:<12}{'ERROR: ' + s['error'][:30]:>34}")
        else:
            print(f"{s['backend']:<12}{_rep_str(s):>16}{_legal_str(s):>18}")
    print("\nPick by the numbers — weight the CLEAN side-on clips most (off-axis ones are "
          "confidence-flagged anyway).")


def main():
    ap = argparse.ArgumentParser(description="Score the pipeline against ground-truth labels")
    ap.add_argument("--pose", default="yolo", choices=list(BACKENDS), help="pose backend to evaluate")
    ap.add_argument("--compare", action="store_true",
                    help="run every backend and print a side-by-side table")
    ap.add_argument("--clips", default=None,
                    help="comma-separated name substrings to include (default: all clips)")
    args = ap.parse_args()

    files = sorted(GT_DIR.glob("*.json"))
    if args.clips:
        wanted = [c.strip() for c in args.clips.split(",")]
        files = [f for f in files if any(w in f.stem for w in wanted)]
    if not files:
        print("No ground-truth files match (see eval/groundtruth/ and eval/README.md).")
        return

    if args.compare:
        scores = []
        for b in BACKENDS:
            try:
                scores.append(score_backend(b, files))
            except Exception as exc:  # a missing backend dep shouldn't kill the whole comparison
                scores.append({"backend": b, "error": str(exc)})
        print_compare(scores)
    else:
        print_single(score_backend(args.pose, files))


if __name__ == "__main__":
    main()
