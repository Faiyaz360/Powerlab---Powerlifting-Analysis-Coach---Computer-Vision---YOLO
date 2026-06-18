"""Strength tier — PER-LIFT DOTS bands (deadlift [0,85,135,185,230], squat [0,80,125,170,215])."""
from src import strength_standards as ss


def test_deadlift_advanced_band():
    t = ss.tier(150, "deadlift")                       # 150 in DL [135, 185) -> Advanced
    assert t["tier"] == "Advanced" and t["idx"] == 2
    assert t["next"] == "Legendary" and t["to_next"] == 35.0   # 185 - 150


def test_same_dots_different_tier_by_lift():
    """The whole point: a given DOTS is harder on a squat, so it tiers higher there."""
    assert ss.tier(220, "deadlift")["tier"] == "Legendary"     # DL ceiling ~270 -> 220 still Legendary
    assert ss.tier(220, "squat")["tier"] == "Godly"            # squat ceiling ~250 -> 220 is Godly


def test_godly_top_of_deadlift_scale():
    t = ss.tier(240, "deadlift")                       # >= 230 -> Godly
    assert t["tier"] == "Godly" and t["idx"] == 4
    assert t["next"] is None and t["to_next"] == 0.0 and t["pct"] == 1.0


def test_strong_gym_deadlift_is_intermediate():
    assert ss.tier(98, "deadlift")["tier"] == "Intermediate"   # ~1.9x-bw DL, harsh by design


def test_unknown_lift_falls_back_to_deadlift():
    assert ss.tier(150, None)["tier"] == ss.tier(150, "deadlift")["tier"]
    assert ss.tier(150, "curl")["tier"] == ss.tier(150, "deadlift")["tier"]


def test_bench_has_its_own_lower_scale():
    """Bench DOTS ceilings are ~165 (vs ~270 deadlift), so bench bands sit far lower."""
    assert ss.tier(150, "bench")["tier"] == "Godly"        # elite bench (>= 140)
    assert ss.tier(150, "deadlift")["tier"] == "Advanced"  # same DOTS is only Advanced on a deadlift
    assert ss.tier(100, "bench")["tier"] == "Advanced"     # bench [85, 115)


def test_total_has_its_own_scale():
    """A 3-lift total uses the separate total bands, not single-lift ones."""
    assert ss.tier(450, "total")["tier"] == "Advanced"         # total [0,300,400,500,600]
    assert ss.tier(450, "deadlift")["tier"] == "Godly"         # same number is off-the-charts for one lift


def test_tier_order_matches_request():
    assert ss.TIERS == ["Beginner", "Intermediate", "Advanced", "Legendary", "Godly"]


def test_missing_dots_is_none():
    assert ss.tier(None, "squat") is None


def test_weight_class_picks_the_class():
    assert ss.weight_class(80, "male") == "83 kg"        # 74 < 80 <= 83
    assert ss.weight_class(74, "male") == "74 kg"        # exactly on the limit
    assert ss.weight_class(82, "female") == "84 kg"


def test_weight_class_superheavy():
    assert ss.weight_class(125, "male") == "120 kg+"
    assert ss.weight_class(90, "female") == "84 kg+"


def test_weight_class_unknown_is_none():
    assert ss.weight_class(None, "male") is None
    assert ss.weight_class(80, "other") is None
