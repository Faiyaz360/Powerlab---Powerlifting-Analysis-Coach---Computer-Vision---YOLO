"""Strength tier — DOTS-band brackets (Beginner -> Godly), harsh cuts [0, 80, 130, 180, 220]."""
from src import strength_standards as ss


def test_advanced_band():
    t = ss.tier(150)                                   # 150 in [130, 180) -> Advanced
    assert t["tier"] == "Advanced" and t["idx"] == 2
    assert t["next"] == "Legendary" and t["to_next"] == 30.0   # 180 - 150


def test_godly_is_top_tier():
    t = ss.tier(230)                                   # >= 220 -> Godly (IPF records ~225-270)
    assert t["tier"] == "Godly" and t["idx"] == 4
    assert t["next"] is None and t["to_next"] == 0.0 and t["pct"] == 1.0


def test_just_below_godly_is_legendary():
    assert ss.tier(210)["tier"] == "Legendary"         # 210 < the 220 Godly cut -> still Legendary


def test_strong_gym_lifter_is_only_intermediate():
    """Harsh by design: a ~98-DOTS lift (a 1.9x-bodyweight deadlift) is Intermediate, not Advanced."""
    assert ss.tier(98)["tier"] == "Intermediate"


def test_beginner_floor():
    t = ss.tier(40)                                    # below the 80 cut
    assert t["tier"] == "Beginner" and t["idx"] == 0


def test_progress_within_band():
    t = ss.tier(100)                                   # inside [80, 130)
    assert t["tier"] == "Intermediate" and 0.0 < t["pct"] < 1.0


def test_tier_order_matches_request():
    assert ss.TIERS == ["Beginner", "Intermediate", "Advanced", "Legendary", "Godly"]


def test_missing_dots_is_none():
    assert ss.tier(None) is None
