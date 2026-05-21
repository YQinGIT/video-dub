"""ASR backend tests — mock (Stage 5), faster-whisper / WhisperX / FunASR.

The `MockASR` tests need no GPU and run anywhere. The faster-whisper, WhisperX
and FunASR tests are marked `@pytest.mark.gpu` and are skipped automatically
when no CUDA device is present (see `conftest.py`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from videodub.asr import ASRBackend, get_asr_backend
from videodub.asr.mock import MockASR
from videodub.config import ASRConfig
from videodub.errors import BackendError
from videodub.schemas import Transcript


def _audio_file(tmp_path: Path) -> Path:
    """An empty placeholder file — MockASR only checks that the path exists."""
    path = tmp_path / "audio.wav"
    path.write_bytes(b"")
    return path


# --------------------------------------------------------------------------- #
# Mock backend — Stage 5; runs on any machine.                                #
# --------------------------------------------------------------------------- #

def test_mock_asr_returns_deterministic_transcript(tmp_path: Path):
    audio = _audio_file(tmp_path)
    first = MockASR().transcribe(audio, ASRConfig(backend="mock"))
    second = MockASR().transcribe(audio, ASRConfig(backend="mock"))

    assert [s.text for s in first.segments] == [s.text for s in second.segments]
    assert first.segments  # the mock transcript is non-empty
    assert all(s.end >= s.start for s in first.segments)


def test_mock_asr_autodetects_chinese(tmp_path: Path):
    out = MockASR().transcribe(_audio_file(tmp_path), ASRConfig(backend="mock"))

    # language is None in config -> the mock "detects" Chinese
    assert out.language == "zh"
    assert out.source_language is None  # ASR output is not a translation


def test_mock_asr_honours_configured_language(tmp_path: Path):
    cfg = ASRConfig(backend="mock", language="ja")
    out = MockASR().transcribe(_audio_file(tmp_path), cfg)

    assert out.language == "ja"


def test_mock_asr_missing_audio_raises(tmp_path: Path):
    with pytest.raises(BackendError, match="not found"):
        MockASR().transcribe(tmp_path / "nope.wav", ASRConfig(backend="mock"))


def test_factory_returns_mock_backend():
    backend = get_asr_backend(ASRConfig(backend="mock"))
    assert isinstance(backend, MockASR)
    assert isinstance(backend, ASRBackend)


# --------------------------------------------------------------------------- #
# faster-whisper backend — Stage 7a; CUDA-bound, skipped without a GPU.        #
# --------------------------------------------------------------------------- #

@pytest.mark.gpu
def test_factory_returns_faster_whisper_backend():
    """The factory builds the real backend cheaply — no model load, no VRAM."""
    from videodub.asr.faster_whisper import FasterWhisperASR

    backend = get_asr_backend(ASRConfig(backend="faster_whisper"))
    assert isinstance(backend, FasterWhisperASR)
    assert isinstance(backend, ASRBackend)


@pytest.mark.gpu
def test_faster_whisper_transcribes_audio(sample_audio: Path):
    """faster-whisper loads on CUDA and maps its output into a `Transcript`.

    Uses the `tiny` model (~75 MB, downloaded on first run) — this is an
    integration smoke test of the CUDA path and the schema mapping, not a
    transcription-accuracy test, so the synthetic tone in `sample_audio` is fine.
    """
    cfg = ASRConfig(backend="faster_whisper", model_size="tiny")
    result = get_asr_backend(cfg).transcribe(sample_audio, cfg)

    assert isinstance(result, Transcript)
    assert all(s.end >= s.start for s in result.segments)


@pytest.mark.gpu
def test_faster_whisper_missing_audio_raises(tmp_path: Path):
    cfg = ASRConfig(backend="faster_whisper", model_size="tiny")
    with pytest.raises(BackendError, match="not found"):
        get_asr_backend(cfg).transcribe(tmp_path / "nope.wav", cfg)


# --------------------------------------------------------------------------- #
# WhisperX backend — Stage 7a; CUDA-bound, skipped without a GPU.              #
# --------------------------------------------------------------------------- #

@pytest.mark.gpu
def test_factory_returns_whisperx_backend():
    """The factory builds the real backend cheaply — no model load, no VRAM."""
    from videodub.asr.whisperx import WhisperXASR

    backend = get_asr_backend(ASRConfig(backend="whisperx"))
    assert isinstance(backend, WhisperXASR)
    assert isinstance(backend, ASRBackend)


@pytest.mark.gpu
def test_whisperx_transcribes_audio(sample_audio: Path):
    """WhisperX loads on CUDA and maps its output into a `Transcript`.

    Uses the `tiny` model — an integration smoke test of the CUDA path, VAD, and
    schema mapping. The synthetic tone in `sample_audio` carries no speech, so
    VAD yields no segments; an empty `Transcript` is the correct result.
    """
    cfg = ASRConfig(backend="whisperx", model_size="tiny")
    result = get_asr_backend(cfg).transcribe(sample_audio, cfg)

    assert isinstance(result, Transcript)
    assert all(s.end >= s.start for s in result.segments)


@pytest.mark.gpu
def test_whisperx_missing_audio_raises(tmp_path: Path):
    cfg = ASRConfig(backend="whisperx", model_size="tiny")
    with pytest.raises(BackendError, match="not found"):
        get_asr_backend(cfg).transcribe(tmp_path / "nope.wav", cfg)


# --------------------------------------------------------------------------- #
# FunASR / Paraformer-zh backend — Mandarin-specialist; CUDA-bound, skipped    #
# without a GPU.                                                               #
# --------------------------------------------------------------------------- #

@pytest.mark.gpu
def test_factory_returns_funasr_backend():
    """The factory builds the real backend cheaply — no model load, no VRAM."""
    from videodub.asr.funasr import FunASRASR

    backend = get_asr_backend(ASRConfig(backend="funasr"))
    assert isinstance(backend, FunASRASR)
    assert isinstance(backend, ASRBackend)


@pytest.mark.gpu
def test_funasr_transcribes_audio(sample_audio: Path):
    """FunASR loads Paraformer on CUDA and maps its output into a `Transcript`.

    An integration smoke test of the CUDA path, VAD, and schema mapping. The
    synthetic tone in `sample_audio` carries no speech, so VAD yields no
    segments; an empty `Transcript` is the correct result. Paraformer is
    Mandarin-only, so the language is always reported as `zh`.
    """
    cfg = ASRConfig(backend="funasr")
    result = get_asr_backend(cfg).transcribe(sample_audio, cfg)

    assert isinstance(result, Transcript)
    assert result.language == "zh"
    assert all(s.end >= s.start for s in result.segments)


@pytest.mark.gpu
def test_funasr_missing_audio_raises(tmp_path: Path):
    cfg = ASRConfig(backend="funasr")
    with pytest.raises(BackendError, match="not found"):
        get_asr_backend(cfg).transcribe(tmp_path / "nope.wav", cfg)
