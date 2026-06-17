"""Build the share caption for an analysed lift — tags @projectfyz with the lift's real numbers.

The annotated video itself is shared straight from the page via the Web Share API (no second clip is
rendered — that would burn extra GPU time for no gain). So this module is just the caption text.
"""
from __future__ import annotations

HANDLE = "@projectfyz"    # owner's IG — tagged in the caption, never watermarked on the video


def _kg(load) -> str:
    return f"{load:g}kg"


def share_caption(meta: dict) -> str:
    """Post caption from the lift's real numbers. Deterministic, tags @projectfyz, ready to paste."""
    lift = (meta.get("lift") or "lift").lower()
    head = f"{lift.capitalize()} @ {_kg(meta['load'])}" if meta.get("load") else lift.capitalize()
    lines = [f"{head} — analysed in PowerLab"]

    facts = []
    legal = meta.get("legal_pass")
    if legal is not None:
        word = "Depth" if lift == "squat" else "Lockout"
        facts.append(f"{word}: {'passed ✅' if legal else 'missed'}")
    if meta.get("score") is not None:
        grade = meta.get("grade")
        facts.append(f"Score {meta['score']}/100" + (f" ({grade})" if grade else ""))
    if facts:
        lines.append(" · ".join(facts))

    if meta.get("peak_ms"):
        lines.append(f"Top bar speed {meta['peak_ms']:g} m/s")

    lines.append(f"AI form breakdown → {HANDLE}")
    tag = lift if lift in ("squat", "deadlift") else "powerlifting"
    lines.append(f"#powerlifting #{tag} #formcheck #VBT #gymtok")
    return "\n".join(lines)
