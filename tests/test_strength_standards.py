"""Strength tier — DOTS-band brackets (Beginner -> Godly), cuts [0, 50, 90, 140, 200]."""
from src import strength_standards as ss


def test_advanced_band():
    t = ss.tier(95)                                    # 95 in [90, 140) -> Advanced
    assert t["tier"] == "Advanced" and t["idx"] == 2
    assert t["next"] == "Legendary" and t["to_next"] == 45.0   # 140 - 95


def test_godly_is_top_tier():
    t = ss.tier(220)                                   # >= 200 -> Godly (real ceiling ~300)
    assert t["tier"] == "Godly" and t["idx"] == 4
    assert t["next"] is None and t["to_next"] == 0.0 and t["pct"] == 1.0


def test_just_below_godly_is_legendary():
    assert ss.tier(150)["tier"] == "Legendary"         # 150 < the 200 Godly cut -> still Legendary


def test_beginner_floor():
    t = ss.tier(20)                                    # below the 50 cut
    assert t["tier"] == "Beginner" and t["idx"] == 0


def test_progress_within_band():
    t = ss.tier(70)                                    # inside [50, 90)
    assert t["tier"] == "Intermediate" and 0.0 < t["pct"] < 1.0


def test_tier_order_matches_request():
    assert ss.TIERS == ["Beginner", "Intermediate", "Advanced", "Legendary", "Godly"]


def test_missing_dots_is_none():
    assert ss.tier(None) is None
