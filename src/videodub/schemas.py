"""Core data contracts.

`Transcript` is the single object that flows ASR -> translation -> (subtitle | TTS).
This module has no dependencies on any stage; every stage depends on it, and stages
never import each other. Keep it that way.
"""

from __future__ import annotations

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
