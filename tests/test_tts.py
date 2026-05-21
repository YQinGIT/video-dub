"""Stage 5 — mock TTS backend tests.

`MockTTS` renders a real audio clip per segment with ffmpeg, so the tests that
exercise it take the `ffmpeg_available` fixture and skip when ffmpeg is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from videodub.config import TTSConfig
from videodub.errors import ConfigError
from videodub.media_io import probe
from videodub.schemas import Segment, SynthesizedAudio, Transcript
from videodub.tts import TTSBackend, get_tts_backend
from videodub.tts.mock import MockTTS

MOCK = TTSConfig(backend="mock")


def _transcript(*spans: tuple[float, float]) -> Transcript:
    """An English transcript with one segment per (start, end) span."""
    segments = [
        Segment(start=start, end=end, text=f"line {i}")
        for i, (start, end) in enumerate(spans)
    ]
    return Transcript(segments=segments, language="en")


def test_mock_tts_renders_one_clip_per_segment(
    ffmpeg_available: None, tmp_path: Path
):
    result = MockTTS().synthesize(_transcript((0.0, 2.0), (2.0, 5.0)), MOCK, tmp_path)

    assert isinstance(result, SynthesizedAudio)
    assert len(result.segments) == 2
    assert all(s.audio_path.exists() for s in result.segments)


def test_mock_tts_clip_duration_matches_segment(
    ffmpeg_available: None, tmp_path: Path
):
    result = MockTTS().synthesize(_transcript((0.0, 3.0)), MOCK, tmp_path)

    clip = result.segments[0]
    assert clip.target_duration == 3.0
    # the mock renders the clip at exactly its slot length
    assert probe(clip.audio_path).duration == pytest.approx(3.0, abs=0.2)


def test_mock_tts_carries_text_and_timestamps(
    ffmpeg_available: None, tmp_path: Path
):
    result = MockTTS().synthesize(_transcript((1.0, 2.5)), MOCK, tmp_path)

    clip = result.segments[0]
    assert (clip.start, clip.end) == (1.0, 2.5)
    assert clip.text == "line 0"


def test_mock_tts_empty_transcript(tmp_path: Path):
    # No segments -> no ffmpeg calls, so this runs even without the binary.
    result = MockTTS().synthesize(Transcript(), MOCK, tmp_path)

    assert result.segments == []
    assert result.sample_rate > 0


def test_factory_returns_mock_backend():
    backend = get_tts_backend(MOCK)
    assert isinstance(backend, MockTTS)
    assert isinstance(backend, TTSBackend)


@pytest.mark.parametrize("backend", ["cosyvoice2", "gpt_sovits", "elevenlabs"])
def test_factory_real_backends_not_implemented(backend: str):
    with pytest.raises(ConfigError, match="not implemented"):
        get_tts_backend(TTSConfig(backend=backend))
