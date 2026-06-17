"""Coach layer: vetted, leashed cues. Tests run the deterministic fallback (no API key, no spend)
and verify the facts the LLM path would receive are fully pre-computed.
"""
import pytest

from src import coach as coachmod
from src.coach import (CLEAN_SET_MSG, COACH_KB, NO_REPS_MSG, _build_facts, generate_cues)
from src.faults import ISSUE_META, detect_faults


@pytest.fixture(autouse=True)
def _force_fallback(monkeypatch):
    # no key -> deterministic fallback in every test (no network, no spend)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


def test_every_fault_has_vetted_sourced_content():
    # leash: no fault may reach a lifter without curated, sourced coaching content
    for issue in ISSUE_META:
        assert issue in COACH_KB, f"{issue} missing from COACH_KB"
        assert COACH_KB[issue]["fix"] and COACH_KB[issue]["source"]


def test_no_reps_message():
    assert generate_cues({"rep_count": 0}, {"issue_list": []}, "squat") == [NO_REPS_MSG]


def test_clean_set_message():
    analysis = {"rep_count": 1, "rep_metrics": [{"ascent_s": 1.0}]}
    assert generate_cues(analysis, {"issue_list": []}, "squat") == [CLEAN_SET_MSG]


def test_facts_precompute_grounds_with_principle_not_the_fix():
    analysis = {
        "rep_count": 2,
        "rep_metrics": [
            {"hips_locked": True, "knees_locked": True, "hip_rise_ratio": 12.25, "ascent_s": 0.8},
            {"hips_locked": True, "knees_locked": True, "hip_rise_ratio": 1.0, "ascent_s": 0.9},
        ],
    }
    faults = detect_faults(analysis, "deadlift")
    facts = _build_facts(analysis, faults, "deadlift")
    hip = next(f for f in facts["detected_faults"] if f["id"] == "hips_rise_too_fast")
    assert hip["principle"] and hip["standard"]   # vetted WHY is sent to ground the model's cue
    assert "fix" not in hip                         # ready-made cue withheld -> the model reasons
    assert hip["numbers"][0]["hip_rise_ratio"] == 12.25     # exact number, no model arithmetic


def test_fallback_wraps_curated_fix_with_opener_and_close():
    cues = coachmod._fallback_cues(["hips_rise_too_fast"], strengths=["all 3 reps locked out"])
    assert cues[0].startswith("Good work")                    # genuine opener from strengths
    assert COACH_KB["hips_rise_too_fast"]["fix"] in cues      # the vetted fix is still delivered
    assert cues[-1] != COACH_KB["hips_rise_too_fast"]["fix"]  # encouraging forward-looking close


def test_strengths_capture_genuine_positives():
    analysis = {"rep_count": 3, "rep_metrics": [
        {"lockout_pass": True, "hips_locked": True, "knees_locked": True, "hip_rise_ratio": 12.25, "ascent_s": 0.8},
        {"lockout_pass": True, "hips_locked": True, "knees_locked": True, "hip_rise_ratio": 1.0, "ascent_s": 0.9},
        {"lockout_pass": True, "hips_locked": True, "knees_locked": True, "hip_rise_ratio": 1.0, "ascent_s": 1.0},
    ]}
    faults = detect_faults(analysis, "deadlift")
    facts = _build_facts(analysis, faults, "deadlift")
    assert facts["strengths"]                                  # passed checks become honest praise
    assert any("legal" in s for s in facts["strengths"])       # all 3 locked out -> a legal strength
