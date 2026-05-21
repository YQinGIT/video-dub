"""Deterministic, offline timing-fit backend.

`MockTimingFitter` assembles the per-segment speech clips into one continuous
vocal track — but it does *no* time-stretching. It simply places each clip at
its segment's start time and lets silence fill the gaps between them.

That works because the clips it is paired with come from `MockTTS`, which
already renders every clip at exactly its segment's duration: no clip overruns
its slot, so none needs stretching. The real backend (Stage 7d, rubberband)
earns its keep on real speech, where clip lengths never line up so neatly.

The placement itself is shared with the rubberband backend — see
`timing._assemble`. Assembling audio needs the ffmpeg binary; it needs no GPU.
"""

from __future__ import annotations

from pathlib import Path

from videodub.config import TimingConfig
from videodub.schemas import SynthesizedAudio
from videodub.timing._assemble import place_on_timeline
from videodub.timing.base import TimingFitter


class MockTimingFitter(TimingFitter):
    """A fake timing fitter: place each clip at its start time, pad gaps, no stretch."""

    def fit(self, synth: SynthesizedAudio, cfg: TimingConfig, out: Path) -> Path:
        # MockTTS renders each clip at exactly its slot length, so no clip
        # overruns and none needs stretching — placement is the whole job.
        clips = [(seg.audio_path, seg.start) for seg in synth.segments]
        return place_on_timeline(clips, synth.sample_rate, out)
