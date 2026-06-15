"""Tests for the per-set interval logic in tools/segment_clips (no video/model needed)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.segment_clips import _active_intervals  # noqa: E402


def _times():
    # 30 samples, one per second
    return np.arange(0, 30, 1.0)


def test_two_sets_split_by_long_rest():
    # Arrange: two bursts (0-3s, 20-24s) with a long rest between
    active = np.zeros(30, dtype=bool)
    active[0:4] = True
    active[20:25] = True
    # Act
    iv = _active_intervals(active, _times(), max_gap=4.0, min_len=2.0, pad=0.0, end_t=30.0)
    # Assert: stays two separate sets
    assert len(iv) == 2


def test_merge_sets_within_short_gap():
    # Two bursts separated by only ~2s of rest -> one merged set
    active = np.zeros(30, dtype=bool)
    active[0:4] = True
    active[6:10] = True
    iv = _active_intervals(active, _times(), max_gap=4.0, min_len=2.0, pad=0.0, end_t=30.0)
    assert len(iv) == 1


def test_drop_short_blip():
    # A single-sample flicker is shorter than min_len -> dropped
    active = np.zeros(30, dtype=bool)
    active[5:6] = True
    iv = _active_intervals(active, _times(), max_gap=4.0, min_len=2.0, pad=0.0, end_t=30.0)
    assert iv == []


def test_padding_is_clamped_to_video_bounds():
    # Burst at 0-3s, pad 1s: start clamps to 0, end becomes 4
    active = np.zeros(30, dtype=bool)
    active[0:4] = True
    iv = _active_intervals(active, _times(), max_gap=4.0, min_len=2.0, pad=1.0, end_t=30.0)
    start, end = iv[0]
    assert start == 0.0
    assert end == 4.0
