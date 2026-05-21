"""Stage 3 — subtitle renderer tests.

These are "golden-string" tests: we build a known `Transcript`, render it, and
compare the output against an exact, hand-written expected string. Subtitle
rendering is deterministic with no randomness, so an exact match is the
strongest check available — any drift in spacing, numbering, or timestamp
spelling fails the test.
"""

import pytest

from videodub.schemas import Segment, Transcript
from videodub.subtitle import render, write

# --------------------------------------------------------------------------- #
# The shared input: a two-segment transcript, reused across the format tests.  #
# "Hello there" runs 0.0 -> 2.0 s; "General Kenobi" runs 2.0 -> 5.5 s.         #
# --------------------------------------------------------------------------- #
TRANSCRIPT = Transcript(
    segments=[
        Segment(start=0.0, end=2.0, text="Hello there"),
        Segment(start=2.0, end=5.5, text="General Kenobi"),
    ],
    language="en",
)


def test_render_srt_exact():
    expected = (
        "1\n"
        "00:00:00,000 --> 00:00:02,000\n"
        "Hello there\n"
        "\n"
        "2\n"
        "00:00:02,000 --> 00:00:05,500\n"
        "General Kenobi\n"
    )
    assert render(TRANSCRIPT, "srt") == expected


def test_render_vtt_exact():
    expected = (
        "WEBVTT\n"
        "\n"
        "00:00:00.000 --> 00:00:02.000\n"
        "Hello there\n"
        "\n"
        "00:00:02.000 --> 00:00:05.500\n"
        "General Kenobi\n"
    )
    assert render(TRANSCRIPT, "vtt") == expected


def test_render_ass_exact():
    expected = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
        "0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
        "Dialogue: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,Hello there\n"
        "Dialogue: 0,0:00:02.00,0:00:05.50,Default,,0,0,0,,General Kenobi\n"
    )
    assert render(TRANSCRIPT, "ass") == expected


def test_render_srt_bilingual_stacks_both_languages():
    """A `secondary` transcript adds a second line under each cue."""
    primary = Transcript(
        segments=[Segment(start=0.0, end=2.0, text="你好")], language="zh"
    )
    secondary = Transcript(
        segments=[Segment(start=0.0, end=2.0, text="Hello")], language="en"
    )
    expected = (
        "1\n"
        "00:00:00,000 --> 00:00:02,000\n"
        "你好\n"      # primary on top
        "Hello\n"    # secondary stacked below
    )
    assert render(primary, "srt", secondary=secondary) == expected


def test_render_ass_bilingual_uses_backslash_n():
    """ASS has no real newline inside a line — the break is the escape \\N."""
    primary = Transcript(segments=[Segment(start=0.0, end=2.0, text="你好")])
    secondary = Transcript(segments=[Segment(start=0.0, end=2.0, text="Hello")])
    out = render(primary, "ass", secondary=secondary)
    assert "Default,,0,0,0,,你好\\NHello\n" in out


def test_timestamps_roll_over_past_one_hour():
    """A segment ending at 3661.5 s must format as 1 h 1 m 1.5 s in every format."""
    long_clip = Transcript(segments=[Segment(start=0.0, end=3661.5, text="x")])
    assert "--> 01:01:01,500" in render(long_clip, "srt")
    assert "--> 01:01:01.500" in render(long_clip, "vtt")
    assert ",1:01:01.50," in render(long_clip, "ass")


def test_empty_transcript_renders_a_valid_but_empty_file():
    """No segments: SRT is empty, VTT/ASS still need their headers."""
    empty = Transcript()
    assert render(empty, "srt") == ""
    assert render(empty, "vtt") == "WEBVTT\n"
    assert render(empty, "ass").startswith("[Script Info]\n")


def test_unknown_format_rejected():
    with pytest.raises(ValueError, match="unknown subtitle format"):
        render(TRANSCRIPT, "txt")  # type: ignore[arg-type]


def test_bilingual_segment_count_mismatch_rejected():
    one = Transcript(segments=[Segment(start=0.0, end=1.0, text="a")])
    two = Transcript(
        segments=[
            Segment(start=0.0, end=1.0, text="a"),
            Segment(start=1.0, end=2.0, text="b"),
        ]
    )
    with pytest.raises(ValueError, match="1:1 segment match"):
        render(one, "srt", secondary=two)


def test_write_infers_format_from_extension(tmp_path):
    out = write(TRANSCRIPT, tmp_path / "subs.vtt")
    assert out.exists()
    assert out.read_text(encoding="utf-8") == render(TRANSCRIPT, "vtt")


def test_write_creates_missing_parent_dirs(tmp_path):
    out = write(TRANSCRIPT, tmp_path / "nested" / "deep" / "subs.srt")
    assert out.exists()


def test_write_unknown_extension_rejected(tmp_path):
    with pytest.raises(ValueError, match="cannot infer subtitle format"):
        write(TRANSCRIPT, tmp_path / "subs.foo")
