"""A/B-test LLM coaching models side by side, using the REAL production prompt + vetted content.

This sends EXACTLY what ships in src/coach.py — the same SYSTEM_PROMPT, the same pre-computed,
leashed facts from coach._build_facts (vetted COACH_KB fix/principle/source per detected fault) —
to several vision-capable models via OpenRouter, with and without keyframes, then prints a
side-by-side comparison of the cues, latency, tokens and cost. So the comparison reflects the
actual coach, not a stand-in prompt.

The honesty leash lives in coach.SYSTEM_PROMPT: a model may ONLY phrase the detected faults.

Usage (set your key in your OWN terminal first, never paste it in chat):
    $env:OPENROUTER_API_KEY = "sk-or-..."
    .\.venv\Scripts\python.exe tools\coach_ab.py --lift deadlift-1 --mode both
    .\.venv\Scripts\python.exe tools\coach_ab.py --dry-run      # build prompt+frames, no API calls
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import cv2
import requests

# Make `src` importable so the A/B uses the SAME prompt + facts builder the app ships.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import coach as coachmod          # noqa: E402  (SYSTEM_PROMPT + _build_facts)
from src.faults import detect_faults       # noqa: E402

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"

# Vision-capable candidates spanning cheap -> flagship. Preflight drops any that don't exist or
# can't read images, so it's safe to list hopefuls — edit freely.
CANDIDATE_MODELS = [
    # cheap tier
    "google/gemini-2.5-flash-lite",    # $0.10/$0.40
    "qwen/qwen3-vl-32b-instruct",      # $0.10/$0.42
    "google/gemini-3.1-flash-lite",    # newest cheap Google
    # mid tier
    "google/gemini-2.5-flash",         # $0.30/$2.50
    "anthropic/claude-haiku-4.5",      # $1/$5 — strong small model, tight leash adherence
    # premium / ceiling — is it worth it for a phrasing task on a leash?
    "google/gemini-3.5-flash",         # $1.50/$9 flagship-class
    "anthropic/claude-sonnet-4.6",     # $3/$15
    "anthropic/claude-opus-4.8",       # $5/$25 — the ceiling
]

MAX_FRAME_HEIGHT = 512   # downscale keyframes to keep image-token cost low
JPEG_QUALITY = 85
MAX_TOKENS = 400
TEMPERATURE = 0.3        # low = consistent, comparable cues

# Experiment prompt for --withhold-fix: the DETECTED fault + numbers are given, but NOT our vetted
# `fix`. The model must reason the cue itself — this tests real coaching knowledge (and whether the
# models finally diverge), while keeping detection deterministic (it still can't invent faults).
WITHHELD_SYSTEM_PROMPT = """You are one of the best powerlifting coaches in the world and a \
certified IPF referee. A deterministic system has already DETECTED which faults occurred on this \
lift and handed you the exact measurements. Your job: for each detected fault, diagnose the cause \
and give your OWN world-class cue to fix it — this is your coaching, not a script.

NON-NEGOTIABLE:
1. Address ONLY the faults in `detected_faults`. Never add, infer, or hint at a fault not in that \
list — the list is final.
2. Use the supplied `numbers` exactly; never invent a figure.
3. Be correct: cue what a top raw-powerlifting coach would actually say for this fault — no generic \
gym-bro advice.

HOW THE BEST COACHES CUE: prioritise the most limiting fault first; one focus at a time (max 3 \
cues); prefer EXTERNAL cues (act on the bar/floor) over internal muscle talk; make it this-rep \
specific with the number; short, confident, plain gym language.

STYLE: talk to the lifter ("you"). 1-2 sentences per cue. A short numbered list, most important \
first. No preamble, no sign-off, no headers, no markdown."""

# Plain-English meaning of each raw measurement, so the model reads the numbers right in
# --full-reasoning (where NO fault is named — the model must find the faults itself).
FIELD_LEGEND = {
    "hip_rise_ratio": "how much faster the hips rose than the shoulders early in the pull "
                      "(1.0 = together; >1 = hips lead)",
    "torso_lean_deg": "trunk angle from vertical at lockout (0 = fully upright)",
    "lockout_hip_angle": "hip angle at the top, deg",
    "lockout_knee_angle": "knee angle at the top, deg (~180 = straight)",
    "hips_locked": "passed the erect-lockout check",
    "knees_locked": "knees straight at the top",
    "lockout_pass": "legal lockout overall",
    "min_knee_angle": "deepest knee bend, deg (squat; lower = deeper, <~100 = below parallel)",
    "max_forward_lean": "peak trunk lean from vertical, deg (squat)",
    "depth_pass": "squat hit legal depth",
    "descent_s": "eccentric (lowering) duration, s",
    "ascent_s": "concentric (lifting) duration, s",
}

# --full-reasoning: the model gets the raw per-rep data (and maybe keyframes) and NO fault list. It
# must diagnose on its own. This is the EXPERIMENT to test whether the AI can find faults reliably
# or whether it hallucinates — its output is not leashed, so judge it for invented faults.
DIAGNOSE_SYSTEM_PROMPT = """You are one of the best powerlifting coaches in the world and a \
certified IPF referee. Below is the full per-rep MEASUREMENT data for one lift (and possibly \
keyframes). Review it like a coach watching film: find any genuine technique or legality faults and \
give specific cues to fix them.

BE RIGOROUSLY HONEST:
- Only call out a fault you can justify FROM THE DATA or clearly SEE in a keyframe — cite the rep \
and the number/observation.
- If the lift looks solid, say so plainly. Do NOT manufacture problems to seem useful.
- Never guess about something you can't measure or see. Say "can't tell from this" instead.

Prioritise the most limiting fault first; max 3 cues; external cues (act on the bar/floor); \
this-rep specific with the number; short, plain gym language. Numbered list, no preamble, no \
headers."""


def load_catalog(candidates: list[str]) -> dict[str, dict]:
    """Preflight OpenRouter: keep only candidates that exist AND accept image input.

    Returns id -> {"prompt": float, "completion": float} per-token USD pricing.
    """
    resp = requests.get(MODELS_URL, timeout=30)
    resp.raise_for_status()
    by_id = {m["id"]: m for m in resp.json()["data"]}

    catalog: dict[str, dict] = {}
    for model_id in candidates:
        info = by_id.get(model_id)
        if info is None:
            print(f"  - skip {model_id}: not on OpenRouter")
            continue
        modalities = info.get("architecture", {}).get("input_modalities", [])
        if "image" not in modalities:
            print(f"  - skip {model_id}: no image input ({modalities})")
            continue
        pricing = info.get("pricing", {})
        catalog[model_id] = {
            "prompt": float(pricing.get("prompt", 0.0) or 0.0),
            "completion": float(pricing.get("completion", 0.0) or 0.0),
        }
    return catalog


def analysis_from_metrics(metrics: dict) -> dict:
    """Rebuild the minimal analysis dict coach/faults need from a saved metrics.json."""
    return {
        "lift": metrics["lift"],
        "rep_count": metrics["rep_count"],
        "rep_metrics": metrics["reps"],
        "bar_velocity": metrics.get("bar_velocity") or [],
    }


def pick_worst_rep(metrics: dict) -> dict:
    """Choose the most instructive rep to show keyframes of."""
    reps = metrics["reps"]
    lift = metrics["lift"]
    if lift == "deadlift":
        return max(reps, key=lambda r: r.get("hip_rise_ratio") or 0.0)
    if lift == "squat":
        return max(reps, key=lambda r: r.get("max_forward_lean") or 0.0)
    return reps[0]


def grab_frame(cap: cv2.VideoCapture, frame_idx: int) -> "cv2.Mat | None":
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
    ok, img = cap.read()
    if not ok:
        return None
    h, w = img.shape[:2]
    if h > MAX_FRAME_HEIGHT:
        scale = MAX_FRAME_HEIGHT / h
        img = cv2.resize(img, (int(w * scale), MAX_FRAME_HEIGHT))
    return img


def extract_keyframes(video: Path, rep: dict, out_dir: Path) -> list[Path]:
    """Save bottom + lockout/depth keyframes of the chosen rep; return their paths."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    bottom_idx = int(round(rep["bottom_s"] * fps))
    top_idx = int(rep.get("badge_frame") or round(rep.get("top_s", rep.get("end_s", 0)) * fps))

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for label, idx in (("bottom", bottom_idx), ("lockout", top_idx)):
        img = grab_frame(cap, idx)
        if img is None:
            continue
        p = out_dir / f"{video.stem}_{label}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        paths.append(p)
    cap.release()
    return paths


def to_data_uri(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/jpeg;base64,{b64}"


def build_messages(facts: dict, frames: list[Path], with_images: bool, system_prompt: str) -> list[dict]:
    # Mirror production (coach._llm_cues) exactly for stats mode; frames mode just attaches images.
    text = "Here is the analysis. Write my coaching cues.\n\n" + json.dumps(facts, indent=2)
    if with_images and frames:
        text += "\n\nAttached: keyframes of the flagged rep (bottom of the lift, then lockout/depth)."
        content = [{"type": "text", "text": text}]
        for p in frames:
            content.append({"type": "image_url", "image_url": {"url": to_data_uri(p)}})
        user = {"role": "user", "content": content}
    else:
        user = {"role": "user", "content": text}
    return [{"role": "system", "content": system_prompt}, user]


def strip_fix(facts: dict) -> dict:
    """Remove our vetted fix/principle/source so the model must reason the cue itself."""
    out = json.loads(json.dumps(facts))   # deep copy
    for f in out.get("detected_faults", []):
        for k in ("fix", "principle", "source"):
            f.pop(k, None)
    return out


def diagnose_facts(metrics: dict) -> dict:
    """Raw per-rep measurements + a field legend, with NO faults named (for --full-reasoning)."""
    drop = {"badge_frame", "badge"}
    reps = [{k: v for k, v in r.items() if k not in drop} for r in metrics["reps"]]
    legend = {k: v for k, v in FIELD_LEGEND.items() if any(k in r for r in reps)}
    return {"lift": metrics["lift"], "rep_count": metrics["rep_count"],
            "reps": reps, "field_legend": legend}


def call_model(key: str, model: str, messages: list[dict], pricing: dict) -> dict:
    """One chat completion. Returns cues text, tokens, cost (USD), latency (s)."""
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "usage": {"include": True},   # ask OpenRouter to return real cost
    }
    headers = {"Authorization": f"Bearer {key}", "X-Title": "PowerLab coach A/B"}
    t0 = time.perf_counter()
    resp = requests.post(OPENROUTER_URL, headers=headers, json=body, timeout=120)
    latency = time.perf_counter() - t0
    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}", "latency": latency}

    data = resp.json()
    text = data["choices"][0]["message"]["content"].strip()
    usage = data.get("usage", {}) or {}
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    cost = usage.get("cost")
    if cost is None:   # fall back to preflight per-token pricing
        cost = pt * pricing["prompt"] + ct * pricing["completion"]
    return {
        "text": text, "prompt_tokens": pt, "completion_tokens": ct,
        "cost": float(cost), "latency": latency,
    }


def render(results: list[dict], facts: dict, frames: list[Path], out_md: Path) -> None:
    """Console table + a readable side-by-side markdown file."""
    det = facts.get("detected_faults")
    if det is None:
        fault_ids = "n/a — full reasoning (model finds faults itself)"
    else:
        fault_ids = [f["id"] for f in det] or "none (clean)"
    print("\n" + "=" * 70)
    print(f"  COACH A/B — {facts['lift']}  |  detected: {fault_ids}")
    print("=" * 70)
    header = f"{'model':<34}{'mode':<8}{'lat(s)':>7}{'tok in/out':>14}{'$/run':>11}{'$/1k':>9}"
    print(header)
    print("-" * len(header))

    lines_md = [
        f"# Coach A/B — {facts['lift']} (production prompt + vetted facts)",
        "",
        f"**Detected faults:** {fault_ids}",
        f"**Keyframes:** {', '.join(p.name for p in frames) or 'none'}",
        "",
        "## Facts sent to every model (incl. vetted fix/principle/source)",
        "```json",
        json.dumps(facts, indent=2),
        "```",
        "",
        "## Cues by model",
        "",
    ]

    for r in results:
        tag = f"{r['model']:<34}{r['mode']:<8}"
        if "error" in r:
            print(tag + f"  ERROR: {r['error'][:60]}")
            lines_md += [f"### {r['model']} ({r['mode']})", "", f"> ERROR: {r['error']}", ""]
            continue
        per_1k = r["cost"] * 1000
        io = f"{r['prompt_tokens']}/{r['completion_tokens']}"
        print(tag + f"{r['latency']:>7.1f}{io:>14}{r['cost']:>11.5f}{per_1k:>9.2f}")
        lines_md += [
            f"### {r['model']} ({r['mode']})",
            "",
            f"*{r['latency']:.1f}s · {r['prompt_tokens']}+{r['completion_tokens']} tok · "
            f"${r['cost']:.5f}/run · ${per_1k:.2f}/1000 runs*",
            "",
            r["text"],
            "",
        ]

    out_md.write_text("\n".join(lines_md), encoding="utf-8")
    print("-" * len(header))
    print(f"  full cues written to: {out_md}")
    print("=" * 70 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="A/B test coach LLMs on real lift data.")
    ap.add_argument("--lift", default="deadlift-1",
                    help="sample stem in output/ (e.g. deadlift-1, squat-1)")
    ap.add_argument("--mode", choices=["stats", "frames", "both"], default="both")
    ap.add_argument("--models", nargs="*", default=CANDIDATE_MODELS)
    ap.add_argument("--withhold-fix", action="store_true",
                    help="strip our vetted fix; the model must reason the cue itself (coaching test)")
    ap.add_argument("--full-reasoning", action="store_true",
                    help="send raw rep data + legend, NO faults named; model must FIND faults itself")
    ap.add_argument("--dry-run", action="store_true", help="build prompt+frames, make no API calls")
    args = ap.parse_args()

    metrics_path = ROOT / "output" / f"{args.lift}_metrics.json"
    video_path = ROOT / "input" / f"{args.lift}.mov"
    if not metrics_path.exists():
        sys.exit(f"missing {metrics_path} — run the pipeline on {args.lift} first")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    analysis = analysis_from_metrics(metrics)
    lift = metrics["lift"]
    faults = detect_faults(analysis, lift)
    facts = coachmod._build_facts(analysis, faults, lift)   # the REAL leashed, vetted facts
    system_prompt = coachmod.SYSTEM_PROMPT
    if args.full_reasoning:
        facts = diagnose_facts(metrics)
        system_prompt = DIAGNOSE_SYSTEM_PROMPT
        print("MODE: --full-reasoning (NO faults named; model must find faults from raw data)")
    elif args.withhold_fix:
        facts = strip_fix(facts)
        system_prompt = WITHHELD_SYSTEM_PROMPT
        print("MODE: --withhold-fix (model reasons the cue; our vetted fix is NOT sent)")

    frames: list[Path] = []
    if args.mode in ("frames", "both") and video_path.exists():
        worst = pick_worst_rep(metrics)
        frames = extract_keyframes(video_path, worst, ROOT / "output" / "ab_frames")
        print(f"keyframes: {[p.name for p in frames]} (rep bottom_s={worst['bottom_s']})")
    elif args.mode in ("frames", "both"):
        print(f"  (no source video {video_path.name}; falling back to stats-only)")

    modes = ["stats", "frames"] if args.mode == "both" else [args.mode]
    modes = [m for m in modes if m == "stats" or frames]   # drop frames mode if no frames

    if args.dry_run:
        print("\n--- DRY RUN: prompt that would be sent ---\n")
        print("SYSTEM:\n" + system_prompt)
        msgs = build_messages(facts, frames, bool(frames), system_prompt)
        user = msgs[1]["content"]
        print("\nUSER:\n" + (user if isinstance(user, str) else user[0]["text"]))
        print(f"\n(+{len(frames)} image(s))" if frames else "\n(no images)")
        print("\nmodels (unvalidated):", args.models)
        return

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit('OPENROUTER_API_KEY not set. In PowerShell: $env:OPENROUTER_API_KEY = "sk-or-..."')

    print("preflight: checking models on OpenRouter...")
    catalog = load_catalog(args.models)
    if not catalog:
        sys.exit("no valid image-capable models from your list")
    print(f"  valid: {list(catalog)}")

    results: list[dict] = []
    for model in catalog:
        for mode in modes:
            with_images = mode == "frames"
            msgs = build_messages(facts, frames, with_images, system_prompt)
            print(f"  calling {model} [{mode}] ...")
            r = call_model(key, model, msgs, catalog[model])
            r.update({"model": model, "mode": mode})
            results.append(r)

    suffix = "_diagnose" if args.full_reasoning else ("_reasoned" if args.withhold_fix else "")
    out_md = ROOT / "output" / f"coach_ab_{args.lift}{suffix}.md"
    render(results, facts, frames, out_md)


if __name__ == "__main__":
    main()
