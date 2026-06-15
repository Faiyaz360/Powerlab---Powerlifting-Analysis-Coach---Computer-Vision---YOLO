"""Gather ONLY Creative-Commons powerlifting clips from YouTube (legally defensible).

Downloads CC-BY videos for the given search terms, dedupes across runs, filters out long
videos, and writes a manifest.csv plus each clip's .info.json as a license audit trail.
Standard-licensed videos are skipped on purpose — this is a COMMERCIAL product (see DATA.md).

Re-running only fetches NEW clips (a download archive remembers what was grabbed), so this
script is safe to schedule (e.g. a weekly Windows Task — see colab/README.md).

Needs yt-dlp:  pip install yt-dlp

Usage:
    # ad-hoc terms
    python tools/gather_clips.py "squat form" "deadlift technique" --max 50
    # or a saved, repeatable query list (recommended)
    python tools/gather_clips.py --queries-file tools/queries.txt --max 40
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

# yt-dlp's per-video license string for CC-BY content
_CC = "Creative Commons Attribution license (reuse allowed)"


def _load_queries(args: argparse.Namespace) -> list[str]:
    """Merge ad-hoc queries + a queries file, drop blanks/comments, keep order, de-dupe."""
    queries = list(args.queries)
    if args.queries_file:
        for line in Path(args.queries_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                queries.append(line)
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique


def _build_manifest(out: Path) -> int:
    """Scan every .info.json into one manifest.csv (the human-readable catalogue + license proof)."""
    rows = []
    for info in sorted(out.glob("*.info.json")):
        d = json.loads(info.read_text(encoding="utf-8"))
        if d.get("_type") == "playlist" or not d.get("id"):
            continue  # skip yt-dlp's playlist-level metadata; only catalogue real videos
        rows.append(
            {
                "id": d.get("id", ""),
                "title": d.get("title", ""),
                "uploader": d.get("uploader", ""),
                "license": d.get("license", ""),
                "duration_s": d.get("duration", ""),
                "url": d.get("webpage_url", ""),
            }
        )
    fields = ["id", "title", "uploader", "license", "duration_s", "url"]
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download CC-licensed lifting clips from YouTube")
    ap.add_argument("queries", nargs="*", help='search terms, e.g. "squat form" deadlift')
    ap.add_argument("--queries-file", help="text file of queries, one per line (# = comment)")
    ap.add_argument("--max", type=int, default=30, help="max results to scan per query")
    ap.add_argument("--max-duration", type=int, default=180, help="skip clips longer than N seconds")
    ap.add_argument("--out", default="clips", help="output directory")
    args = ap.parse_args()

    if shutil.which("yt-dlp") is None:
        print("yt-dlp not found. Install it first:  pip install yt-dlp")
        return

    queries = _load_queries(args)
    if not queries:
        print("No queries given. Pass terms or use --queries-file tools/queries.txt")
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "downloaded.txt"  # yt-dlp's cross-run dedup ledger
    # CC-BY only AND short enough to be a single set, not a 20-min compilation
    match = f"license = '{_CC}' & duration < {args.max_duration}"

    for q in queries:
        print(f"\n=== CC-only: {q} ===")
        cmd = [
            "yt-dlp",
            "--match-filter", match,             # skip non-CC and over-long videos
            "--download-archive", str(archive),  # skip anything already downloaded
            "--write-info-json",                 # keep the license audit trail
            "-f", "mp4/best",
            "-o", str(out / "%(id)s.%(ext)s"),
            f"ytsearch{args.max}:{q}",
        ]
        subprocess.run(cmd, check=False)

    n = _build_manifest(out)
    print(f"\nDone. {n} clips catalogued in {out}/manifest.csv (+ per-clip .info.json).")
    print("Keep every .info.json — it proves the CC license at download time (DATA.md).")


if __name__ == "__main__":
    main()
