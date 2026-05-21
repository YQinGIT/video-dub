"""The ASR stage's backend contract.

An `ASRBackend` turns an audio file into a timed `Transcript` — the first object
in the pipeline, the thing every later stage refines. The real backend (Stage
7a, faster-whisper) is CUDA-bound; the mock backend is not, so the whole
pipeline stays testable on any machine.

This module imports only `schemas` and `config` — never `torch` — so
`import videodub.asr` stays cheap.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from videodub.config import ASRConfig
from videodub.schemas import Transcript


class ASRBackend(ABC):
    """A swappable speech-recognition backend, selected by `ASRConfig.backend`."""

    @abstractmethod
    def transcribe(self, audio: Path, cfg: ASRConfig) -> Transcript:
        """Transcribe the speech in `audio` into a timed `Transcript`."""
