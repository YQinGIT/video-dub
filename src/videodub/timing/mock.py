"""Deterministic, offline timing-fit backend.

`MockTimingFitter` assembles the per-segment speech clips into one continuous
vocal track — but it does *no* time-stretching. It simply places each clip at
its segment's start time and lets silence fill the gaps between them.

That works because the clips it is paired with come from `MockTTS`, which
already renders every clip at exactly its segment's duration: no clip overruns
its slot, so none needs stretching. The real backend (Stage 7d) earns its keep
on real speech, where clip lengths never line up so neatly.

Assembling audio needs the ffmpeg binary; it needs no GPU.
"""

from __future__ import annotations

from pathlib import Path

from videodub._ffmpeg import make_silence, run_ffmpeg
from videodub.config import TimingConfig
from videodub.schemas import SynthesizedAudio
from videodub.timing.base import TimingFitter


class MockTimingFitter(TimingFitter):
    """A fake timing fitter: place each clip at its start time, pad gaps, no stretch."""

    def fit(self, synth: SynthesizedAudio, cfg: TimingConfig, out: Path) -> Path:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)

        # No clips -> an empty (zero-length) silent track. Nothing to place.
        if not synth.segments:
            return make_silence(out, duration=0.0, sample_rate=synth.sample_rate)

        # One ffmpeg input per clip; `adelay` shifts clip i to its start time.
        inputs: list[str] = []
        delays: list[tuple[int, int]] = []  # (input index, delay in ms)
        for index, seg in enumerate(synth.segments):
            inputs += ["-i", str(seg.audio_path)]
            delays.append((index, max(round(seg.start * 1000), 0)))

        if len(delays) == 1:
            # `amix` needs at least two inputs; a single delayed clip *is* the
            # whole track, so delay it straight into the output label.
            index, delay_ms = delays[0]
            filtergraph = f"[{index}:a]adelay={delay_ms}:all=1[out]"
        else:
            # Delay each clip, then sum the delayed streams. Because the clips
            # never overlap, summing just lays them side by side on the
            # timeline and the gaps stay silent. `amix` runs until its longest
            # input ends, so the track lasts until the final clip's slot closes.
            parts: list[str] = []
            labels: list[str] = []
            for index, delay_ms in delays:
                label = f"d{index}"
                parts.append(f"[{index}:a]adelay={delay_ms}:all=1[{label}]")
                labels.append(f"[{label}]")
            parts.append(
                "".join(labels)
                + f"amix=inputs={len(labels)}:normalize=0:dropout_transition=0[out]"
            )
            filtergraph = ";".join(parts)

        run_ffmpeg(
            [
                *inputs,
                "-filter_complex", filtergraph,
                "-map", "[out]",
                "-ac", "1",
                "-ar", str(synth.sample_rate),
                str(out),
            ]
        )
        return out
