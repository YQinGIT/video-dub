"""Stage 5 — mock separation backend tests.

`MockSeparator` writes real audio files, so it needs the ffmpeg binary. Tests
that exercise it take the `sample_audio` fixture, which skips them when ffmpeg
is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from videodub.config import SeparationConfig
from videodub.errors import BackendError, ConfigError
from videodub.media_io import probe
from videodub.schemas import SeparatedAudio
from videodub.separation import Separator, get_separator
from videodub.separation.mock import MockSeparator

MOCK = SeparationConfig(backend="mock")


def test_mock_separation_produces_two_files(sample_audio: Path, tmp_path: Path):
    result = MockSeparator().separate(sample_audio, MOCK, tmp_path)

    assert isinstance(result, SeparatedAudio)
    assert result.vocals.exists()
    assert result.background.exists()


def test_mock_separation_vocals_is_exact_copy(sample_audio: Path, tmp_path: Path):
    result = MockSeparator().separate(sample_audio, MOCK, tmp_path)

    assert result.vocals.read_bytes() == sample_audio.read_bytes()


def test_mock_separation_background_matches_duration(
    sample_audio: Path, tmp_path: Path
):
    result = MockSeparator().separate(sample_audio, MOCK, tmp_path)

    source = probe(sample_audio)
    background = probe(result.background)
    assert background.duration == pytest.approx(source.duration, abs=0.2)


def test_mock_separation_missing_audio_raises(tmp_path: Path):
    # The existence check fires before any ffmpeg call, so this needs no binary.
    with pytest.raises(BackendError, match="not found"):
        MockSeparator().separate(tmp_path / "nope.wav", MOCK, tmp_path)


def test_factory_returns_mock_backend():
    backend = get_separator(MOCK)
    assert isinstance(backend, MockSeparator)
    assert isinstance(backend, Separator)


def test_factory_demucs_not_implemented():
    with pytest.raises(ConfigError, match="not implemented"):
        get_separator(SeparationConfig(backend="demucs"))
