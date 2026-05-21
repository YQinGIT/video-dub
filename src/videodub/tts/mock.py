"""Deterministic, offline TTS backend.

`MockTTS` does not really synthesize speech — it has no voice model. For each
segment it writes a sine tone lasting exactly the segment's duration, so the
`SynthesizedAudio` it returns has the right shape and timing for every later
stage to work on. A tone (rather than silence) means the final mixed audio is
actually audible, which makes a mock end-to-end run easy to sanity-check.

Generating audio needs the ffmpeg binary; it needs no GPU and no network.
"""

from __future__ import annotations

from pathlib import Path

from videodub._ffmpeg import make_tone
from videodub.config import TTSConfig
from videodub.schemas import SynthesizedAudio, SynthSegment, Transcript
from videodub.tts.base import TTSBackend

# CosyVoice 2 — the eventual real default — renders at 24 kHz; the mock matches
# it so swapping in the real backend later changes no sample rate downstream.
_SAMPLE_RATE = 24000
# A fixed, low tone. Same for every clip — this is a stand-in, not a melody.
_FREQUENCY = 220.0


class MockTTS(TTSBackend):
    """A fake TTS backend: one fixed-pitch sine tone per segment."""

    def synthesize(
        self, transcript: Transcript, cfg: TTSConfig, out_dir: Path
    ) -> SynthesizedAudio:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        segments: list[SynthSegment] = []
        for index, seg in enumerate(transcript.segments):
            clip = out_dir / f"segment_{index:04d}.wav"
            make_tone(
                clip,
                duration=seg.duration,
                sample_rate=_SAMPLE_RATE,
                frequency=_FREQUENCY,
            )
            segments.append(
                SynthSegment(
                    start=seg.start,
                    end=seg.end,
                    audio_path=clip,
                    text=seg.text,
                    speaker=seg.speaker,
                )
            )

        return SynthesizedAudio(segments=segments, sample_rate=_SAMPLE_RATE)
