"""Deterministic, offline source-separation backend.

`MockSeparator` does not really separate anything — with no model it cannot
tell speech from music. It fakes the split in the cheapest faithful way:

  * `vocals` is a byte-for-byte copy of the input;
  * `background` is a silent track of the same duration and sample rate.

So "vocals + background" still reconstructs something the length of the input,
and every later stage receives the two files it expects.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from videodub._ffmpeg import run_ffmpeg
from videodub.config import SeparationConfig
from videodub.errors import BackendError
from videodub.schemas import SeparatedAudio
from videodub.separation.base import Separator


class MockSeparator(Separator):
    """A fake separator: vocals = input copy, background = matching silence."""

    def separate(
        self, audio: Path, cfg: SeparationConfig, out_dir: Path
    ) -> SeparatedAudio:
        audio = Path(audio)
        if not audio.exists():
            raise BackendError(f"audio file not found: {audio}")

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Vocals: an exact copy. Keep the input's extension so its format is
        # unchanged — no re-encode, nothing to get wrong.
        vocals = out_dir / f"vocals{audio.suffix}"
        shutil.copyfile(audio, vocals)

        # Background: silence. The `volume=0` filter zeroes every sample while
        # leaving duration and sample rate exactly as the input — so we get a
        # matching silent track without having to probe the file first.
        background = out_dir / "background.wav"
        run_ffmpeg(["-i", str(audio), "-af", "volume=0", str(background)])

        return SeparatedAudio(vocals=vocals, background=background)
