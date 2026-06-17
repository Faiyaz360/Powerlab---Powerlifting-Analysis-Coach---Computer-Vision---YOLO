"""Browser-codec transcode decision (pure)."""
from src import media


def test_browser_safe_codec_passes_through():
    assert media._needs_transcode("h264", "yuv420p") is False


def test_hevc_needs_transcode():
    assert media._needs_transcode("hevc", "yuv420p10le") is True


def test_10bit_h264_still_needs_transcode():
    assert media._needs_transcode("h264", "yuv420p10le") is True


def test_unknown_codec_left_alone():
    assert media._needs_transcode(None, None) is False


def test_4k_portrait_needs_downscale():
    # iPhone 4K portrait: long side 3840 > 1280.
    assert media._needs_downscale(2160, 3840) is True


def test_landscape_1080p_needs_downscale():
    # 1920x1080: long side 1920 > 1280.
    assert media._needs_downscale(1920, 1080) is True


def test_at_target_not_downscaled():
    # 540x960: long side == 960 (TARGET_LONG_SIDE), not greater -> leave it.
    assert media._needs_downscale(540, 960) is False


def test_unknown_size_left_alone():
    assert media._needs_downscale(0, 0) is False
