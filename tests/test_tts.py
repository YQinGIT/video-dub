"""TTS backend tests — mock (Stage 5) and IndexTTS-2 (Stage 7c).

`MockTTS` renders a real audio clip per segment with ffmpeg, so the tests that
exercise it take the `ffmpeg_available` fixture and skip when ffmpeg is absent.

IndexTTS-2 runs in its *own* isolated venv, driven as a subprocess, so the
backend module imports cheaply — its wiring (factory, pre-flight guards) is
tested without a GPU or the install. The one end-to-end test that drives the
real venv is `@pytest.mark.gpu` and skips unless IndexTTS-2 is installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from videodub._ffmpeg import run_ffmpeg
from videodub.config import TTSConfig
from videodub.errors import BackendError, ConfigError
from videodub.media_io import probe
from videodub.schemas import Segment, SynthesizedAudio, SynthSegment, Transcript
from videodub.tts import TTSBackend, get_tts_backend
from videodub.tts.mock import MockTTS
from videodub.tts.silence import strip_silence, trim_silence

MOCK = TTSConfig(backend="mock")


def _transcript(*spans: tuple[float, float]) -> Transcript:
    """An English transcript with one segment per (start, end) span."""
    segments = [
        Segment(start=start, end=end, text=f"line {i}")
        for i, (start, end) in enumerate(spans)
    ]
    return Transcript(segments=segments, language="en")


# --------------------------------------------------------------------------- #
# Mock backend — Stage 5; runs on any machine.                                #
# --------------------------------------------------------------------------- #

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


def test_mock_tts_ignores_reference_audio(ffmpeg_available: None, tmp_path: Path):
    """The mock has no voice model, so a reference clip changes nothing."""
    result = MockTTS().synthesize(
        _transcript((0.0, 1.0)),
        MOCK,
        tmp_path,
        reference_audio=tmp_path / "does_not_exist.wav",
    )

    assert len(result.segments) == 1


def test_mock_tts_empty_transcript(tmp_path: Path):
    # No segments -> no ffmpeg calls, so this runs even without the binary.
    result = MockTTS().synthesize(Transcript(), MOCK, tmp_path)

    assert result.segments == []
    assert result.sample_rate > 0


def test_factory_returns_mock_backend():
    backend = get_tts_backend(MOCK)
    assert isinstance(backend, MockTTS)
    assert isinstance(backend, TTSBackend)


@pytest.mark.parametrize("backend", ["gpt_sovits", "elevenlabs"])
def test_factory_unimplemented_backends_raise(backend: str):
    with pytest.raises(ConfigError, match="not implemented"):
        get_tts_backend(TTSConfig(backend=backend))


# --------------------------------------------------------------------------- #
# IndexTTS-2 backend — Stage 7c. IndexTTS-2 runs in its own isolated venv and  #
# is driven out of process, so the backend module imports cheaply and these   #
# wiring tests need neither a GPU nor the IndexTTS-2 install.                  #
# --------------------------------------------------------------------------- #

def test_factory_returns_indextts2_backend():
    from videodub.tts.indextts2 import IndexTTS2TTS

    backend = get_tts_backend(TTSConfig(backend="indextts2"))
    assert isinstance(backend, IndexTTS2TTS)
    assert isinstance(backend, TTSBackend)


def test_indextts2_empty_transcript(tmp_path: Path):
    """An empty transcript returns early — no subprocess, no install needed."""
    from videodub.tts.indextts2 import IndexTTS2TTS

    result = IndexTTS2TTS().synthesize(
        Transcript(), TTSConfig(backend="indextts2"), tmp_path
    )
    assert result.segments == []
    assert result.sample_rate > 0


def test_indextts2_missing_reference_raises(tmp_path: Path):
    """A non-empty transcript with no reference fails before any subprocess."""
    from videodub.tts.indextts2 import IndexTTS2TTS

    with pytest.raises(BackendError, match="reference"):
        IndexTTS2TTS().synthesize(
            _transcript((0.0, 1.0)), TTSConfig(backend="indextts2"), tmp_path
        )


def test_indextts2_not_installed_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Pointed at a directory with no IndexTTS-2 venv, the backend says so."""
    from videodub.tts.indextts2 import IndexTTS2TTS

    monkeypatch.setenv("VIDEODUB_INDEXTTS_HOME", str(tmp_path / "no-install"))
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"")  # exists -> passes the reference check
    with pytest.raises(BackendError, match="not installed"):
        IndexTTS2TTS().synthesize(
            _transcript((0.0, 1.0)),
            TTSConfig(backend="indextts2"),
            tmp_path,
            reference_audio=reference,
        )


# --------------------------------------------------------------------------- #
# Silence trimming — the TTS post-process that strips dead air from each clip  #
# before the timing stage. Portable: needs only the ffmpeg binary.            #
# --------------------------------------------------------------------------- #

def _gapped_clip(path: Path) -> None:
    """Write a 3.3s WAV: 0.3s silence, 1s tone, 1s silence, 1s tone.

    The leading silence and the long internal gap are exactly the dead air the
    trimmer must remove — it stands in for a padded neural-TTS clip.
    """
    run_ffmpeg(
        [
            "-f", "lavfi", "-i", "sine=frequency=300:sample_rate=22050:duration=1",
            "-f", "lavfi", "-i", "sine=frequency=300:sample_rate=22050:duration=1",
            "-filter_complex",
            "[0:a]adelay=300:all=1[a];[1:a]adelay=2300:all=1[b];"
            "[a][b]amix=inputs=2:normalize=0[out]",
            "-map", "[out]", "-ac", "1",
            str(path),
        ]
    )


def test_strip_silence_removes_dead_air(ffmpeg_available: None, tmp_path: Path):
    src = tmp_path / "padded.wav"
    _gapped_clip(src)
    assert probe(src).duration == pytest.approx(3.3, abs=0.2)

    dst = strip_silence(src, tmp_path / "trimmed.wav")

    # Leading silence is gone and the 1s internal gap is capped to a short
    # pause, so what remains is the ~2s of tone plus one small gap.
    assert dst.exists()
    assert 1.8 < probe(dst).duration < 2.7


def test_trim_silence_shortens_clips_and_keeps_metadata(
    ffmpeg_available: None, tmp_path: Path
):
    src = tmp_path / "clip.wav"
    _gapped_clip(src)
    synth = SynthesizedAudio(
        segments=[
            SynthSegment(
                start=2.0, end=4.0, audio_path=src, text="hello", speaker="A"
            )
        ],
        sample_rate=22050,
    )

    result = trim_silence(synth, tmp_path / "_trimmed")

    seg = result.segments[0]
    assert seg.audio_path != src  # rewritten to a fresh, trimmed file
    assert seg.audio_path.exists()
    assert probe(seg.audio_path).duration < probe(src).duration
    # The slot timestamps, text and speaker are carried over untouched.
    assert (seg.start, seg.end, seg.text, seg.speaker) == (2.0, 4.0, "hello", "A")
    assert result.sample_rate == 22050


def test_trim_silence_empty_is_a_noop(tmp_path: Path):
    # No segments -> no ffmpeg calls, so this runs even without the binary.
    result = trim_silence(SynthesizedAudio(segments=[], sample_rate=22050), tmp_path)

    assert result.segments == []
    assert result.sample_rate == 22050


@pytest.mark.gpu
def test_indextts2_synthesizes_via_subprocess(tmp_path: Path):
    """End-to-end: drive the real IndexTTS-2 venv and get a WAV back.

    Skipped unless IndexTTS-2 is actually installed (its venv and a real voice
    reference both present). This exercises the subprocess path; synthesis
    quality is a human-listen check, per the Stage 7 plan.
    """
    from videodub.tts.indextts2 import IndexTTS2TTS, _resolve_home

    home = _resolve_home()
    if not (home / ".venv" / "bin" / "python").exists():
        pytest.skip("IndexTTS-2 venv not installed")
    # A real voice clip — filter out git-LFS pointer stubs by size.
    refs = [
        p for p in sorted((home / "examples").glob("*.wav"))
        if p.stat().st_size > 2048
    ]
    if not refs:
        pytest.skip("no real IndexTTS-2 example voice available")

    transcript = Transcript(
        segments=[Segment(start=0.0, end=3.0, text="Hello, this is a test.")],
        language="en",
    )
    result = IndexTTS2TTS().synthesize(
        transcript, TTSConfig(backend="indextts2"), tmp_path, reference_audio=refs[0]
    )
    assert len(result.segments) == 1
    assert result.segments[0].audio_path.exists()
    assert result.sample_rate > 0
