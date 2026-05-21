"""The separation stage's backend contract.

A `Separator` splits one audio track into two: isolated `vocals` and the
`background` (music, sound effects, room tone). The real backend (Stage 7b,
Demucs) is CUDA-bound; the mock backend is not.

This module imports only `schemas` and `config` — never `torch` — so
`import videodub.separation` stays cheap.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from videodub.config import SeparationConfig
from videodub.schemas import SeparatedAudio


class Separator(ABC):
    """A swappable source-separation backend, selected by `SeparationConfig.backend`."""

    @abstractmethod
    def separate(
        self, audio: Path, cfg: SeparationConfig, out_dir: Path
    ) -> SeparatedAudio:
        """Split `audio` into vocal and background files written under `out_dir`."""
