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

from videodub.config import TTSConfig
from videodub.errors import BackendError, ConfigError
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
