"""The TTS stage's backend contract.

A `TTSBackend` turns a translated `Transcript` into `SynthesizedAudio` — one
rendered speech clip per segment. Real backends (Stage 7c, CosyVoice 2) clone
the original speaker's voice and are CUDA-bound; the mock backend is not.

The output artifact (`SynthSegment` / `SynthesizedAudio`) lives in `schemas`,
not here: the `timing` stage consumes it, and stages communicate only through
`schemas` — never by importing one another.

This module imports only `schemas` and `config` — never `torch` — so
`import videodub.tts` stays cheap.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from videodub.config import TTSConfig
from videodub.schemas import SynthesizedAudio, Transcript


class TTSBackend(ABC):
    """A swappable speech-synthesis backend, selected by `TTSConfig.backend`."""

    @abstractmethod
    def synthesize(
        self, transcript: Transcript, cfg: TTSConfig, out_dir: Path
    ) -> SynthesizedAudio:
        """Render every segment of `transcript` to a speech clip under `out_dir`."""
