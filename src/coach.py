"""Coaching layer: issue list -> plain-English cues.

Phase 1 is rule-based templates. Phase 5 adds an optional Claude path that ONLY rephrases the
same issue list (never adds faults), falling back to these templates when no API key is set.
"""
from __future__ import annotations

RULE_CUES = {
    # squat
    "depth_fail": (
        "Depth not reached — in competition (IPF) this is a red light. Sink until your hip "
        "crease drops below the top of your knee. A slightly wider stance or some ankle/hip "
        "mobility work usually helps."
    ),
    "excessive_forward_lean": (
        "Your torso tips forward a lot coming out of the hole. Brace harder and drive your "
        "upper back into the bar to keep your chest up and the bar over midfoot."
    ),
    # deadlift
    "hips_not_locked": (
        "You're not standing fully erect at the top — in competition the shoulders must come "
        "back. Drive your hips through and squeeze your glutes to finish tall."
    ),
    "knees_not_locked": (
        "Your knees aren't locked straight at the top — that's a red light. Finish the rep by "
        "fully straightening the knees before lowering the bar."
    ),
    "hips_rise_too_fast": (
        "Your hips shoot up faster than your shoulders off the floor, turning it into a "
        "stiff-legged pull. Push the floor away with your legs first and keep the chest up so "
        "hips and shoulders rise together."
    ),
    # shared
    "inconsistent_tempo": (
        "Your rep tempo is uneven across the set — control each rep and keep a steady pace."
    ),
}

NO_REPS_MSG = (
    "No clear reps detected. Check the camera is side-on, your whole body and the bar are in "
    "frame, and the lighting is decent."
)

CLEAN_SET_MSG = "Solid set — everything we measure looks good. Keep it up."


def coach_from_issues(issue_list, rep_count: int):
    if rep_count == 0:
        return [NO_REPS_MSG]
    if not issue_list:
        return [CLEAN_SET_MSG]
    return [RULE_CUES.get(issue, f"Issue detected: {issue}") for issue in issue_list]
