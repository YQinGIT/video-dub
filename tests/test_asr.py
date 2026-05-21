"""Stage 5 — mock ASR backend tests.

`MockASR` needs no GPU and no ffmpeg: it returns a fixed transcript. So every
test here runs on any machine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from videodub.asr import ASRBackend, get_asr_backend
from videodub.asr.mock import MockASR
from videodub.config import ASRConfig
from videodub.errors import BackendError, ConfigError


def _audio_file(tmp_path: Path) -> Path:
    """An empty placeholder file — MockASR only checks that the path exists."""
    path = tmp_path / "audio.wav"
    path.write_bytes(b"")
    return path


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


def test_factory_faster_whisper_not_implemented():
    with pytest.raises(ConfigError, match="not implemented"):
        get_asr_backend(ASRConfig(backend="faster_whisper"))
