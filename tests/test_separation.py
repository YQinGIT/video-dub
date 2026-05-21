"""Separation backend tests — mock (Stage 5) and Demucs (Stage 7b).

`MockSeparator` writes real audio files, so it needs the ffmpeg binary. Tests
that exercise it take the `sample_audio` fixture, which skips them when ffmpeg
is absent. The Demucs tests are marked `@pytest.mark.gpu` and are skipped
automatically when no CUDA device is present (see `conftest.py`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from videodub.config import SeparationConfig
from videodub.errors import BackendError
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


# --------------------------------------------------------------------------- #
# Demucs backend — Stage 7b; CUDA-bound, skipped without a GPU.                #
# --------------------------------------------------------------------------- #

@pytest.mark.gpu
def test_factory_returns_demucs_backend():
    """The factory builds the real backend cheaply — no model load, no VRAM."""
    from videodub.separation.demucs import DemucsSeparator

    backend = get_separator(SeparationConfig(backend="demucs"))
    assert isinstance(backend, DemucsSeparator)
    assert isinstance(backend, Separator)


@pytest.mark.gpu
def test_demucs_separates_audio(sample_audio: Path, tmp_path: Path):
    """Demucs loads on CUDA and writes a full-length vocal / background split.

    Uses `htdemucs` (one ~80 MB model) rather than the default `htdemucs_ft`
    (a bag of four) — this is an integration smoke test of the CUDA path and
    the stem-to-schema mapping, not a separation-quality test, so the synthetic
    tone in `sample_audio` is fine.
    """
    cfg = SeparationConfig(backend="demucs", model="htdemucs")
    result = get_separator(cfg).separate(sample_audio, cfg, tmp_path)

    assert isinstance(result, SeparatedAudio)
    assert result.vocals.exists()
    assert result.background.exists()

    source = probe(sample_audio)
    for stem in (result.vocals, result.background):
        assert probe(stem).duration == pytest.approx(source.duration, abs=0.2)


@pytest.mark.gpu
def test_demucs_missing_audio_raises(tmp_path: Path):
    cfg = SeparationConfig(backend="demucs", model="htdemucs")
    with pytest.raises(BackendError, match="not found"):
        get_separator(cfg).separate(tmp_path / "nope.wav", cfg, tmp_path)
