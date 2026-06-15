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
