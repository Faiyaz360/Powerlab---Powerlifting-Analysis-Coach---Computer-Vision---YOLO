"""Coaching layer: turn the deterministic fault list into plain-English cues.

The pipeline DETECTS faults (faults.py); this module only PHRASES them. It never invents a fault.

Two paths:
  * Rule-based fallback (no API key) — returns the curated ``fix`` text per fault directly.
  * LLM reasoning (OPENROUTER_API_KEY set) — an OpenAI-compatible model writes its OWN cue for each
    DETECTED fault, grounded by the vetted ``principle`` + PRE-COMPUTED numbers (the ready-made
    ``fix`` is withheld so it coaches rather than parrots). It never does arithmetic and never
    addresses anything outside the detected list. Any error falls back to the rule path.

Why this split: an A/B (tools/coach_ab.py) showed letting the model FIND faults makes it hallucinate
(it flagged legal 162° lockouts as "soft"), but letting it reason the CUE for a fault we already
detected is both safe and genuinely better coaching. So detection stays deterministic; phrasing is
the model's.

Every cue traces to a real coaching source (COACH_KB ``source`` field) — IPF rules for legal
faults, recognised strength sources for coaching faults. That is how "verified coaching" is
enforced here: in our curated content, not in the model's guesses.
"""
from __future__ import annotations

import json
import os
import re

import requests

from .faults import ISSUE_META

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"   # override via COACH_MODEL env var

NO_REPS_MSG = (
    "No clear reps detected. Check the camera is side-on, your whole body and the bar are in "
    "frame, and the lighting is decent."
)
CLEAN_SET_MSG = "Solid set — everything we measure looks good. Keep it up."

# Vetted, sourced coaching content per fault id. `fix` = the actionable cue (self-contained, used
# verbatim by the keyless fallback); `principle` = the vetted WHY (sent to the LLM to ground its own
# cue); `source` = where it's grounded. The LLM reasons its own cue from `principle` + the numbers.
COACH_KB = {
    "depth_fail": {
        "fix": "Hit depth — your hip crease has to drop below the top of your knee or it's a red "
               "light. Sit straight down to full depth; a slightly wider stance plus ankle and hip "
               "mobility usually unlocks it.",
        "principle": "Squat depth is the most common competition red light; controlled depth beats "
                     "grinding a high rep.",
        "source": "IPF Technical Rules 2026 (squat depth); Stronger By Science squat guide.",
    },
    "excessive_forward_lean": {
        "fix": "Brace harder — big breath into the belly, ribs down — and drive your upper back "
               "into the bar to keep your chest up and the bar stacked over midfoot.",
        "principle": "Excessive trunk lean moves the bar ahead of midfoot and shifts load to the "
                     "lower back, costing you drive out of the hole.",
        "source": "Rippetoe, Starting Strength 3e (bar over midfoot); Barbell Medicine squat.",
    },
    "hips_not_locked": {
        "fix": "Finish fully erect — squeeze your glutes and push your hips through until your "
               "shoulders are back over the bar. Don't hitch or ramp it up.",
        "principle": "IPF requires a clearly erect finish with the shoulders back; an unlocked hip "
                     "is a red light even on a heavy pull.",
        "source": "IPF Technical Rules 2026 (deadlift lockout: erect, shoulders back).",
    },
    "knees_not_locked": {
        "fix": "Lock your knees straight at the top and hold until the down command — soft knees "
               "at lockout are a red light.",
        "principle": "Full knee extension is a required part of a legal deadlift lockout.",
        "source": "IPF Technical Rules 2026 (deadlift: knees locked).",
    },
    "hips_rise_too_fast": {
        "fix": "Off the floor, push the floor away with your legs and keep your chest up so your "
               "hips and shoulders rise together — leading with the hips turns it into a "
               "stiff-legged pull and rounds your back.",
        "principle": "When the hips shoot up first the bar stalls on the floor while the back angle "
                     "steepens, raising spinal load; chest and hips should rise as one.",
        "source": "Rippetoe, Starting Strength (hip/shoulder timing); Barbell Medicine deadlift.",
    },
}

SYSTEM_PROMPT = """You are one of the best powerlifting coaches in the world and a certified IPF \
referee. A deterministic system has DETECTED the faults and what went well, with exact numbers. \
Turn it into feedback that makes the lifter better AND want to come back.

WHAT YOU'RE GIVEN:
- `strengths`: what genuinely went well (passed checks) — use these for honest praise; never invent \
praise beyond them.
- `detected_faults`: each with its `principle` (the vetted why) and `numbers`.

RULES (non-negotiable):
1. Address ONLY the faults in `detected_faults`. Never add, infer, or hint at a fault not listed — \
detection is done; if something seems off but isn't listed, it passed. Stay silent on it.
2. Use the `numbers` exactly; never invent or recompute a figure.
3. Coach each fault's `principle` — turn it into an actionable EXTERNAL cue (act on the bar/floor, \
not muscle talk); don't just restate the principle.

STRUCTURE (evidence-based — NOT a praise/criticism/praise "sandwich"; lifters see through that):
1. OPEN with ONE genuine, specific positive drawn from `strengths` (truthful, not flattery). One \
sentence.
2. Then the FIX: lead with the single most important fault; at most TWO cues (coach one thing at a \
time). Each: name the rep + the number, then the external cue. Frame it as an opportunity to add \
kilos, not a failure.
3. CLOSE with one short, encouraging, forward-looking line — confidence they'll own it next set. \
NOT another compliment.

TONE: motivating and autonomy-supportive ("try…", "chase…", "next set…"), lean positive, plain gym \
language, like a coach standing next to them. Direct and honest, but never harsh or insulting.

FORMAT: the opener line, then a short numbered list for the fix(es), then the close line. Each cue \
1-2 short sentences. No headers, no bold, no markdown."""

# rep_metrics fields to surface per per-rep fault, so the model phrases real numbers (no math).
_FAULT_FIELDS = {
    "depth_fail": ("min_knee_angle",),
    "excessive_forward_lean": ("max_forward_lean",),
    "hips_not_locked": ("lockout_hip_angle", "torso_lean_deg"),
    "knees_not_locked": ("lockout_knee_angle",),
    "hips_rise_too_fast": ("hip_rise_ratio",),
}


def generate_cues(analysis: dict, faults: dict, lift: str) -> list[str]:
    """Cues for a lift. LLM-phrased when OPENROUTER_API_KEY is set, else the vetted rule fallback.

    Never raises and never blocks: any LLM/network error degrades to the deterministic fallback.
    """
    if not analysis.get("rep_count"):
        return [NO_REPS_MSG]
    issue_list = faults.get("issue_list", [])
    if not issue_list:
        return [CLEAN_SET_MSG]

    facts = _build_facts(analysis, faults, lift)
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        try:
            cues = _llm_cues(facts, key)
            if cues:
                return cues
        except Exception:
            pass  # network/parse/API error -> deterministic fallback below
    return _fallback_cues(issue_list, facts.get("strengths"))


def _build_facts(analysis: dict, faults: dict, lift: str) -> dict:
    """Pre-computed, leashed facts: every detected fault + its vetted content + exact numbers."""
    rep_metrics = analysis.get("rep_metrics") or []
    detected = []
    for issue in faults["issue_list"]:
        meta = ISSUE_META.get(issue, {})
        kb = COACH_KB.get(issue, {})
        entry = {
            "id": issue,
            "category": meta.get("category"),
            "standard": meta.get("rule"),
            # The vetted WHY grounds the cue's accuracy. The ready-made `fix` is WITHHELD so the
            # model reasons its own cue (it's kept in COACH_KB for the keyless fallback).
            "principle": kb.get("principle"),
        }
        reps = _offending_reps(issue, faults)
        fields = _FAULT_FIELDS.get(issue, ())
        entry["reps_affected"] = reps
        entry["numbers"] = [
            {"rep": rp, **{f: rep_metrics[rp - 1].get(f) for f in fields}}
            for rp in reps if 0 <= rp - 1 < len(rep_metrics)
        ]
        detected.append(entry)
    return {"lift": lift, "rep_count": analysis.get("rep_count"),
            "strengths": _strengths(analysis, faults, lift), "detected_faults": detected}


def _offending_reps(issue: str, faults: dict) -> list[int]:
    """1-based rep numbers that carry this issue (per-rep faults only)."""
    return [pr["rep"] for pr in faults.get("per_rep", []) if issue in pr.get("issues", [])]


def _strengths(analysis: dict, faults: dict, lift: str) -> list[str]:
    """Deterministic 'what genuinely went well', so the coach can open with honest praise (not
    flattery). Built from the same passed checks the pipeline already computed."""
    rep_metrics = analysis.get("rep_metrics") or []
    n = len(rep_metrics)
    out = []
    legal_key = "depth_pass" if lift == "squat" else "lockout_pass"
    legal = sum(1 for r in rep_metrics if r.get(legal_key))
    legal_word = "hit depth" if lift == "squat" else "locked out"
    if n and legal == n:
        out.append(f"all {n} reps were legal ({legal_word} per IPF)")
    elif legal:
        out.append(f"{legal} of {n} reps were legal ({legal_word})")
    # reps with no coaching faults on them
    faulty = {pr["rep"] for pr in faults.get("per_rep", []) if pr.get("issues")}
    clean = [i for i in range(1, n + 1) if i not in faulty]
    if clean and len(clean) == n:
        out.append("technique was clean across the set")
    elif clean:
        out.append(", ".join(f"rep {i}" for i in clean) + " had clean technique")
    return out


def _llm_cues(facts: dict, key: str) -> list[str]:
    """Phrase the vetted facts via an OpenAI-compatible model (OpenRouter). May raise on error."""
    user = "Here is the analysis. Write my coaching cues.\n\n" + json.dumps(facts, indent=2)
    body = {
        "model": os.environ.get("COACH_MODEL", DEFAULT_MODEL),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "max_tokens": 400,
        "temperature": 0.3,
    }
    headers = {"Authorization": f"Bearer {key}", "X-Title": "PowerLab coach"}
    resp = requests.post(OPENROUTER_URL, headers=headers, json=body, timeout=60)
    resp.raise_for_status()
    return _split_cues(resp.json()["choices"][0]["message"]["content"])


def _split_cues(text: str) -> list[str]:
    """Split a model's numbered/bulleted reply into individual cue strings."""
    cues = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line.strip())
        if line:
            cues.append(line)
    return cues


def _fallback_cues(issue_list: list[str], strengths: list[str] | None = None) -> list[str]:
    """Deterministic cues (no model): a genuine opener from `strengths`, the curated `fix` per
    detected fault, then an encouraging close — the same motivating shape as the LLM path."""
    out = []
    if strengths:
        out.append(f"Good work — {strengths[0]}.")
    for issue in issue_list:
        kb = COACH_KB.get(issue)
        out.append(kb["fix"] if kb else f"Work on: {issue.replace('_', ' ')}.")
    out.append("Lock in that fix and chase it next set — you've got this.")
    return out


def coach_from_issues(issue_list, rep_count: int) -> list[str]:
    """Back-compat wrapper (rule-based only). Prefer generate_cues for the full LLM-capable path."""
    if rep_count == 0:
        return [NO_REPS_MSG]
    if not issue_list:
        return [CLEAN_SET_MSG]
    return _fallback_cues(issue_list)
