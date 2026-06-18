"""Strength tier — DOTS-band brackets (Beginner -> Godly)."""
from src import strength_standards as ss


def test_advanced_band():
    t = ss.tier(95)                                    # 95 in [80, 110) -> Advanced
    assert t["tier"] == "Advanced" and t["idx"] == 2
    assert t["next"] == "Legendary" and t["to_next"] == 15.0   # 110 - 95


def test_godly_is_top_tier():
    t = ss.tier(150)                                   # >= 140 -> Godly
    assert t["tier"] == "Godly" and t["idx"] == 4
    assert t["next"] is None and t["to_next"] == 0.0 and t["pct"] == 1.0


def test_beginner_floor():
    t = ss.tier(20)                                    # below the 50 cut
    assert t["tier"] == "Beginner" and t["idx"] == 0


def test_progress_within_band():
    t = ss.tier(65)                                    # halfway through [50, 80)
    assert t["tier"] == "Intermediate" and t["pct"] == 0.5


def test_tier_order_matches_request():
    assert ss.TIERS == ["Beginner", "Intermediate", "Advanced", "Legendary", "Godly"]


def test_missing_dots_is_none():
    assert ss.tier(None) is None
