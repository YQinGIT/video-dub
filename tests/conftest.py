"""Shared pytest fixtures.

pytest imports this file automatically; fixtures defined here are visible to
every test in `tests/` without an explicit import.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

# media_io and the mock audio backends are portable but still need the ffmpeg
# binaries. When they are absent we skip those tests rather than fail the suite.
_HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny real .mp4 (1 s, 160x120, with an audio tone), built once per run.

    `testsrc` and `sine` are ffmpeg's built-in synthetic sources, so no media
    file needs to be committed to the repo.
    """
    if not _HAS_FFMPEG:
        pytest.skip("ffmpeg/ffprobe not on PATH")

    path = tmp_path_factory.mktemp("media") / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=160x120:rate=15",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-c:v", "libx264",
            "-c:a", "aac",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return path


@pytest.fixture(scope="session")
def sample_audio(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny real .wav (1 s, 16 kHz mono, 440 Hz tone), built once per run.

    Input for the mock `separation` backend, and for any other test that needs
    a real audio file to work on.
    """
    if not _HAS_FFMPEG:
        pytest.skip("ffmpeg/ffprobe not on PATH")

    path = tmp_path_factory.mktemp("audio") / "sample.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f", "lavfi",
            "-i", "sine=frequency=440:duration=1:sample_rate=16000",
            "-ac", "1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return path


@pytest.fixture
def ffmpeg_available() -> None:
    """Skip the requesting test when ffmpeg is not on PATH.

    The mock `tts` and `timing` backends fabricate audio with ffmpeg but take
    no audio file as input, so they cannot lean on `sample_audio` for the skip.
    """
    if not _HAS_FFMPEG:
        pytest.skip("ffmpeg not on PATH")
