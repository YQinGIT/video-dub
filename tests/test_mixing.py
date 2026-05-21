"""Stage 6 — mixing stage tests.

`mix` shells out to ffmpeg, so the tests take the `ffmpeg_available` fixture and
build their input tracks with the `_ffmpeg` helpers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from videodub._ffmpeg import make_silence, make_tone
from videodub.config import MixingConfig
from videodub.errors import BackendError
from videodub.media_io import probe
from videodub.mixing import mix


def test_mix_combines_two_tracks(ffmpeg_available: None, tmp_path: Path):
    vocals = make_tone(tmp_path / "v.wav", duration=2.0, sample_rate=24000)
    background = make_tone(
        tmp_path / "b.wav", duration=3.0, sample_rate=16000, frequency=110.0
    )
    out = mix(vocals, background, tmp_path / "mixed.wav", MixingConfig())

    assert out.exists()
    # amix runs to the longer of the two inputs -> 3 s
    assert probe(out).duration == pytest.approx(3.0, abs=0.2)


def test_mix_resamples_mismatched_inputs(ffmpeg_available: None, tmp_path: Path):
    # 24 kHz vocals + 16 kHz background — mix must resample, not crash
    vocals = make_tone(tmp_path / "v.wav", duration=1.0, sample_rate=24000)
    background = make_silence(tmp_path / "b.wav", duration=1.0, sample_rate=16000)
    out = mix(vocals, background, tmp_path / "mixed.wav", MixingConfig())

    assert probe(out).audio_streams[0].sample_rate == 48000


def test_mix_missing_input_raises(ffmpeg_available: None, tmp_path: Path):
    background = make_silence(tmp_path / "b.wav", duration=1.0, sample_rate=16000)
    with pytest.raises(BackendError, match="vocals audio not found"):
        mix(tmp_path / "nope.wav", background, tmp_path / "out.wav", MixingConfig())
