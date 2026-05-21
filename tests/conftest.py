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

# The rubberband timing backend shells out to the `rubberband` binary; absent
# it, the rubberband tests skip rather than fail the suite.
_HAS_RUBBERBAND = bool(shutil.which("rubberband"))


def _cuda_available() -> bool:
    """True when CTranslate2 can see a CUDA device.

    CTranslate2 is the inference engine behind the faster-whisper ASR backend and
    is installed only with the `gpu` extra; on a portable machine the import
    fails, so the GPU-marked tests are skipped instead of erroring.
    """
    try:
        import ctranslate2
    except ImportError:
        return False
    try:
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


_HAS_CUDA = _cuda_available()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip every `@pytest.mark.gpu` test when no CUDA device is available.

    Keeps `uv run pytest` green and CUDA-free on any machine; the real GPU
    backends are exercised only where a device actually exists.
    """
    if _HAS_CUDA:
        return
    skip = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


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


@pytest.fixture
def rubberband_available() -> None:
    """Skip the requesting test when the `rubberband` binary is not on PATH.

    The rubberband timing backend needs both ffmpeg and the Rubber Band CLI, so
    a test exercising it takes this fixture alongside `ffmpeg_available`.
    """
    if not _HAS_RUBBERBAND:
        pytest.skip("rubberband not on PATH")
