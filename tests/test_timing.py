"""Stage 5 — mock timing-fit backend tests.

`MockTimingFitter` assembles real audio with ffmpeg. Its input is the output of
`MockTTS`, so these tests build that first; both need ffmpeg, hence the
`ffmpeg_available` fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from videodub.config import TimingConfig, TTSConfig
from videodub.errors import ConfigError
from videodub.media_io import probe
from videodub.schemas import Segment, SynthesizedAudio, Transcript
from videodub.timing import TimingFitter, get_timing_fitter
from videodub.timing.mock import MockTimingFitter
from videodub.tts.mock import MockTTS

MOCK = TimingConfig(backend="mock")


def _synthesized(tmp_path: Path, *spans: tuple[float, float]) -> SynthesizedAudio:
    """Run MockTTS over the given (start, end) spans to get real input clips."""
    segments = [
        Segment(start=start, end=end, text=f"line {i}")
        for i, (start, end) in enumerate(spans)
    ]
    transcript = Transcript(segments=segments, language="en")
    return MockTTS().synthesize(transcript, TTSConfig(backend="mock"), tmp_path)


def test_mock_timing_assembles_continuous_track(
    ffmpeg_available: None, tmp_path: Path
):
    synth = _synthesized(tmp_path, (0.0, 2.0), (2.0, 5.0))
    out = MockTimingFitter().fit(synth, MOCK, tmp_path / "vocals.wav")

    assert out.exists()
    # back-to-back clips -> the track lasts to the final segment's end
    assert probe(out).duration == pytest.approx(5.0, abs=0.2)


def test_mock_timing_pads_gaps_with_silence(ffmpeg_available: None, tmp_path: Path):
    # a 2 s clip at 0-2, a silent gap, then a 1 s clip at 4-5
    synth = _synthesized(tmp_path, (0.0, 2.0), (4.0, 5.0))
    out = MockTimingFitter().fit(synth, MOCK, tmp_path / "vocals.wav")

    # the silent 2-4 gap is kept -> the track runs the full 5 s
    assert probe(out).duration == pytest.approx(5.0, abs=0.2)


def test_mock_timing_single_segment(ffmpeg_available: None, tmp_path: Path):
    synth = _synthesized(tmp_path, (1.0, 3.0))
    out = MockTimingFitter().fit(synth, MOCK, tmp_path / "vocals.wav")

    assert out.exists()
    # one clip starting at 1 s and running 2 s -> the track ends at 3 s
    assert probe(out).duration == pytest.approx(3.0, abs=0.2)


def test_mock_timing_empty_input(ffmpeg_available: None, tmp_path: Path):
    synth = SynthesizedAudio(segments=[], sample_rate=24000)
    out = MockTimingFitter().fit(synth, MOCK, tmp_path / "vocals.wav")

    assert out.exists()


def test_factory_returns_mock_backend():
    backend = get_timing_fitter(MOCK)
    assert isinstance(backend, MockTimingFitter)
    assert isinstance(backend, TimingFitter)


def test_factory_rubberband_not_implemented():
    with pytest.raises(ConfigError, match="not implemented"):
        get_timing_fitter(TimingConfig(backend="rubberband"))
