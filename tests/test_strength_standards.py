"""Strength-tier standards: bodyweight-ratio brackets per lift + sex."""
from src import strength_standards as ss


def test_male_squat_2x_bodyweight_is_advanced():
    t = ss.tier(200, 100, "male", "squat")            # ratio 2.0 -> Advanced (Elite is 2.5x)
    assert t["tier"] == "Advanced" and t["idx"] == 3
    assert t["next"] == "Elite"
    assert t["to_next_kg"] == 50.0                     # 2.5*100 - 200
    assert 0.0 <= t["pct"] <= 1.0


def test_elite_has_no_next_and_full_progress():
    t = ss.tier(300, 100, "male", "deadlift")          # ratio 3.0 -> above the 2.75x Elite cut
    assert t["tier"] == "Elite" and t["idx"] == 4
    assert t["next"] is None and t["to_next_kg"] == 0.0 and t["pct"] == 1.0


def test_below_beginner_clamps_to_beginner():
    t = ss.tier(40, 100, "male", "squat")              # ratio 0.4 -> below the 1.0x floor
    assert t["tier"] == "Beginner" and t["idx"] == 0


def test_female_uses_its_own_ratios():
    """A load that is 'Advanced' for a woman would be lower-tier for a man — sex matters."""
    t = ss.tier(105, 70, "female", "squat")            # ratio 1.5 -> Advanced (female squat 1.5x)
    assert t["tier"] == "Advanced"


def test_unknown_sex_or_lift_is_none():
    assert ss.tier(150, 80, "other", "squat") is None
    assert ss.tier(150, 80, "male", "bench") is None   # bench standards not defined yet


def test_missing_inputs_are_none():
    assert ss.tier(None, 80, "male", "squat") is None
    assert ss.tier(150, 0, "male", "squat") is None
