"""Timing-fit backend tests — mock (Stage 5) and rubberband (Stage 7d).

Both backends assemble real audio with ffmpeg, so every test takes the
`ffmpeg_available` fixture. The rubberband backend also shells out to the
`rubberband` binary, so its tests additionally take `rubberband_available`.

`timing` is portable — rubberband is an ordinary system package, not a GPU
library — so these are plain tests, not `@pytest.mark.gpu`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from videodub._ffmpeg import make_tone
from videodub.config import TimingConfig, TTSConfig
from videodub.errors import BackendError
from videodub.media_io import probe
from videodub.schemas import Segment, SynthesizedAudio, SynthSegment, Transcript
from videodub.timing import TimingFitter, get_timing_fitter
from videodub.timing.mock import MockTimingFitter
from videodub.timing.rubberband import RubberbandTimingFitter
from videodub.tts.mock import MockTTS

MOCK = TimingConfig(backend="mock")
RUBBERBAND = TimingConfig(backend="rubberband")
_SR = 22050


def _synthesized(tmp_path: Path, *spans: tuple[float, float]) -> SynthesizedAudio:
    """Run MockTTS over the given (start, end) spans to get real input clips.

    MockTTS renders each clip at exactly its slot length — the already-fits
    case the mock fitter is designed for.
    """
    segments = [
        Segment(start=start, end=end, text=f"line {i}")
        for i, (start, end) in enumerate(spans)
    ]
    transcript = Transcript(segments=segments, language="en")
    return MockTTS().synthesize(transcript, TTSConfig(backend="mock"), tmp_path)


def _mismatched(
    tmp_path: Path, *clips: tuple[float, tuple[float, float]]
) -> SynthesizedAudio:
    """Build SynthesizedAudio whose clip *files* deliberately mis-fit their slots.

    Each `clips` entry is `(clip_seconds, (slot_start, slot_end))`: a real tone
    of `clip_seconds` paired with a segment whose target is that slot. MockTTS
    always renders an exact fit, so this hand-builds the mismatch the rubberband
    fitter exists to correct.
    """
    segments = []
    for index, (clip_len, (start, end)) in enumerate(clips):
        path = make_tone(
            tmp_path / f"clip_{index}.wav", duration=clip_len, sample_rate=_SR
        )
        segments.append(
            SynthSegment(start=start, end=end, audio_path=path, text=f"line {index}")
        )
    return SynthesizedAudio(segments=segments, sample_rate=_SR)


# --------------------------------------------------------------------------- #
# Mock backend — Stage 5; runs anywhere ffmpeg does.                          #
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Rubberband backend — Stage 7d; portable, needs the `rubberband` binary.     #
# --------------------------------------------------------------------------- #

def test_factory_returns_rubberband_backend():
    """The factory builds the backend cheaply — no binary needed to select it."""
    backend = get_timing_fitter(RUBBERBAND)
    assert isinstance(backend, RubberbandTimingFitter)
    assert isinstance(backend, TimingFitter)


def test_rubberband_stretches_short_clip_to_slot(
    ffmpeg_available: None, rubberband_available: None, tmp_path: Path
):
    # a 1.0 s clip that must fill a 1.2 s slot -> ratio 1.2, within max 1.3
    synth = _mismatched(tmp_path, (1.0, (0.0, 1.2)))
    out = RubberbandTimingFitter().fit(synth, RUBBERBAND, tmp_path / "vocals.wav")

    assert probe(out).duration == pytest.approx(1.2, abs=0.12)


def test_rubberband_compresses_long_clip_to_slot(
    ffmpeg_available: None, rubberband_available: None, tmp_path: Path
):
    # a 1.2 s clip in a 1.0 s slot -> ratio 0.83, within min 0.7
    synth = _mismatched(tmp_path, (1.2, (0.0, 1.0)))
    out = RubberbandTimingFitter().fit(synth, RUBBERBAND, tmp_path / "vocals.wav")

    assert probe(out).duration == pytest.approx(1.0, abs=0.12)


def test_rubberband_clamps_excessive_stretch(
    ffmpeg_available: None, rubberband_available: None, tmp_path: Path
):
    # a 1.0 s clip in a 3.0 s slot would need 3x; max_stretch caps it at 1.3, so
    # the clip reaches only ~1.3 s — it is not stretched all the way to the slot.
    synth = _mismatched(tmp_path, (1.0, (0.0, 3.0)))
    out = RubberbandTimingFitter().fit(synth, RUBBERBAND, tmp_path / "vocals.wav")

    assert probe(out).duration == pytest.approx(1.3, abs=0.15)


def test_rubberband_trims_clip_that_overruns_after_clamping(
    ffmpeg_available: None, rubberband_available: None, tmp_path: Path
):
    # a 3.0 s clip in a 1.0 s slot would need 0.33x; min_stretch floors it at
    # 0.7, leaving ~2.1 s — still over the slot, so it is trimmed back to 1.0 s.
    synth = _mismatched(tmp_path, (3.0, (0.0, 1.0)))
    out = RubberbandTimingFitter().fit(synth, RUBBERBAND, tmp_path / "vocals.wav")

    assert probe(out).duration == pytest.approx(1.0, abs=0.1)


def test_rubberband_pads_gap_between_clips(
    ffmpeg_available: None, rubberband_available: None, tmp_path: Path
):
    # both clips already fit their slots; a silent 1-2 s gap sits between them
    synth = _mismatched(tmp_path, (1.0, (0.0, 1.0)), (1.0, (2.0, 3.0)))
    out = RubberbandTimingFitter().fit(synth, RUBBERBAND, tmp_path / "vocals.wav")

    assert probe(out).duration == pytest.approx(3.0, abs=0.15)


def test_rubberband_skips_stretch_when_clip_already_fits(
    ffmpeg_available: None, rubberband_available: None, tmp_path: Path
):
    # clip length == slot length -> ratio 1.0, so rubberband is never invoked
    # and no stretched intermediate is written.
    synth = _mismatched(tmp_path, (1.0, (0.0, 1.0)))
    out = tmp_path / "vocals.wav"
    RubberbandTimingFitter().fit(synth, RUBBERBAND, out)

    assert out.exists()
    assert not (out.parent / "_rubberband" / "stretch_0000.wav").exists()


def test_rubberband_empty_input(ffmpeg_available: None, tmp_path: Path):
    # no clips -> an empty silent track, no rubberband call
    synth = SynthesizedAudio(segments=[], sample_rate=_SR)
    out = RubberbandTimingFitter().fit(synth, RUBBERBAND, tmp_path / "vocals.wav")

    assert out.exists()


def test_rubberband_rejects_invalid_stretch_bounds(
    ffmpeg_available: None, tmp_path: Path
):
    # min_stretch > max_stretch makes clamping nonsensical -> a clear error
    synth = _mismatched(tmp_path, (1.0, (0.0, 1.0)))
    bad = TimingConfig(backend="rubberband", min_stretch=1.5, max_stretch=1.3)
    with pytest.raises(BackendError, match="stretch bounds"):
        RubberbandTimingFitter().fit(synth, bad, tmp_path / "vocals.wav")
