"""The timing stage's backend contract.

A `TimingFitter` takes `SynthesizedAudio` — the loose speech clips the `tts`
stage rendered — and assembles them into one continuous vocal track, fitting
each clip into the slot its segment's timestamps mark out. The real backend
(Stage 7d, rubberband) time-stretches each clip to fit; the mock backend does
no stretching at all.

`timing` is CPU-only, but the real backend shells out to the `rubberband`
binary. This module imports only `schemas` and `config`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from videodub.config import TimingConfig
from videodub.schemas import SynthesizedAudio


class TimingFitter(ABC):
    """A swappable timing-fit backend, selected by `TimingConfig.backend`."""

    @abstractmethod
    def fit(self, synth: SynthesizedAudio, cfg: TimingConfig, out: Path) -> Path:
        """Assemble `synth` into one continuous vocal track written to `out`."""
