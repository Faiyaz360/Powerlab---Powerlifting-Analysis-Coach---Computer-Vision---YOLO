"""Local dev entry point.

Usage:
    python cli.py input/squat.mp4
    python cli.py input/squat.mp4 --lift squat --out output
"""
from __future__ import annotations

import argparse

from src.pipeline import analyze


def main() -> None:
    ap = argparse.ArgumentParser(description="Powerlifting form analysis (squat / deadlift)")
    ap.add_argument("video", help="path to a side-on lift video")
    ap.add_argument("--lift", default="squat", choices=["squat", "deadlift"], help="lift type")
    ap.add_argument("--pose", default="yolo", choices=["mediapipe", "yolo", "rtmpose"],
                    help="pose backend: yolo (default), rtmpose (Apache, cloud GPU), mediapipe (CPU)")
    ap.add_argument("--plate", default="hsv", choices=["hsv", "yolo"],
                    help="plate detector: hsv (default) or yolo (trained, colour-agnostic)")
    ap.add_argument("--out", default="output", help="output directory")
    args = ap.parse_args()

    print(f"Analyzing {args.video} (pose={args.pose}, plate={args.plate}) ...")
    result = analyze(args.video, lift=args.lift, out_dir=args.out, backend=args.pose,
                     plate_backend=args.plate)

    print(f"\nReps detected: {result['rep_count']}")
    print("Coaching:")
    for cue in result["cues"]:
        print(f"  - {cue}")
    print("\nOutputs:")
    for key, value in result["paths"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
