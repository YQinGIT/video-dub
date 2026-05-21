"""Shared pytest fixtures.

pytest imports this file automatically; fixtures defined here are visible to
every test in `tests/` without an explicit import.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

# media_io is portable but still needs the ffmpeg binaries. When they are
# absent we skip the media tests rather than fail the suite.
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
