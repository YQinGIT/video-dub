"""Core data contracts.

`Transcript` is the object that flows ASR -> translation -> (subtitle | TTS).
Past TTS the pipeline carries audio rather than text, so two more contracts
join it: `SeparatedAudio` (the vocal / background split) and `SynthesizedAudio`
(the rendered dub clips).

This module depends on no stage; every stage depends on it, and stages never
import each other — so every cross-stage artifact is defined right here. Keep
it that way.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class Segment(BaseModel):
    """One timed unit of speech — the atom of a Transcript."""

    start: float = Field(ge=0, description="Start time, seconds")
    end: float = Field(ge=0, description="End time, seconds")
    text: str = Field(description="Text for this segment")
    speaker: str | None = Field(default=None, description="Diarization label, if any")

    @property
    def duration(self) -> float:
        """Length of the segment in seconds."""
        return self.end - self.start

    @model_validator(mode="after")
    def _ordered(self) -> Segment:
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) precedes start ({self.start})")
        return self


class Transcript(BaseModel):
    """Ordered timed segments.

    Translation produces a *new* Transcript with the same timestamps and translated
    `text`; it is contractually 1:1, so list index aligns source <-> translated
    segments.
    """

    segments: list[Segment] = Field(default_factory=list)
    language: str | None = Field(
        default=None, description="ISO code of the text currently in `segments`"
    )
    source_language: str | None = Field(
        default=None, description="Original language, set when this is a translation"
    )

    @property
    def duration(self) -> float:
        """End time of the last segment, or 0.0 when empty."""
        return self.segments[-1].end if self.segments else 0.0


class SeparatedAudio(BaseModel):
    """The `separation` stage's output: one audio track split into two files.

    `vocals` carries the isolated speech — it feeds ASR and, later, voice
    cloning. `background` is everything else (music, sound effects, room tone);
    it is preserved untouched and remixed under the finished dub.
    """

    vocals: Path = Field(description="Isolated speech track")
    background: Path = Field(description="Everything except speech — music, SFX")


class SynthSegment(BaseModel):
    """One synthesized speech clip, aligned 1:1 with a translated `Segment`.

    `start` and `end` are the slot this clip should occupy on the dubbed
    timeline — copied straight from the translated `Segment`. `audio_path`
    points at the clip a TTS backend actually rendered; its real length on disk
    may not match `end - start`, and closing that gap is the `timing` stage's
    job. So these timestamps are the *target*, not a measurement of the file.
    """

    start: float = Field(ge=0, description="Target start time, seconds")
    end: float = Field(ge=0, description="Target end time, seconds")
    audio_path: Path = Field(description="The rendered speech clip on disk")
    text: str = Field(description="The text spoken in this clip")
    speaker: str | None = Field(default=None, description="Speaker label, if any")

    @property
    def target_duration(self) -> float:
        """The timeline slot, in seconds, this clip must be fitted to."""
        return self.end - self.start

    @model_validator(mode="after")
    def _ordered(self) -> SynthSegment:
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) precedes start ({self.start})")
        return self


class SynthesizedAudio(BaseModel):
    """The `tts` stage's output: every rendered speech clip, in order.

    This is deliberately *not* a `Transcript` — its segments carry audio files,
    not text to translate further. `sample_rate` is the rate every clip shares,
    recorded once so downstream stages need not probe each file.
    """

    segments: list[SynthSegment] = Field(default_factory=list)
    sample_rate: int = Field(gt=0, description="Sample rate shared by every clip, Hz")

    @property
    def duration(self) -> float:
        """End time of the last clip's slot, or 0.0 when empty."""
        return self.segments[-1].end if self.segments else 0.0
