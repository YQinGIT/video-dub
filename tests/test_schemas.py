"""Stage 1 — Segment / Transcript contract tests."""

import pytest
from pydantic import ValidationError

from videodub.schemas import Segment, Transcript


def test_segment_basic_and_duration():
    seg = Segment(start=1.0, end=3.5, text="hello")
    assert seg.duration == 2.5
    assert seg.speaker is None


def test_segment_zero_duration_is_allowed():
    seg = Segment(start=2.0, end=2.0, text="")
    assert seg.duration == 0.0


def test_segment_end_before_start_rejected():
    with pytest.raises(ValidationError, match="precedes start"):
        Segment(start=5.0, end=3.0, text="bad")


def test_segment_negative_time_rejected():
    with pytest.raises(ValidationError):
        Segment(start=-1.0, end=1.0, text="bad")


def test_transcript_empty_duration_is_zero():
    assert Transcript().duration == 0.0
    assert Transcript().segments == []


def test_transcript_duration_is_last_segment_end():
    t = Transcript(
        segments=[
            Segment(start=0.0, end=2.0, text="one"),
            Segment(start=2.0, end=5.0, text="two"),
        ],
        language="zh",
    )
    assert t.duration == 5.0
    assert t.language == "zh"
    assert t.source_language is None


def test_transcript_roundtrips_through_json():
    t = Transcript(
        segments=[Segment(start=0.0, end=1.0, text="hi", speaker="S1")],
        language="en",
        source_language="zh",
    )
    assert Transcript.model_validate_json(t.model_dump_json()) == t
