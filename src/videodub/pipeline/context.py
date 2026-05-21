"""The artifact bag carried through one pipeline run.

A `PipelineContext` is created by the runner before a recipe starts and handed
to every stage in turn. Each stage reads the artifacts it needs and writes the
ones it produces; the artifact fields begin as None and fill in as the recipe
advances. Stages never call each other — the context is their only channel.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from videodub.config import Settings
from videodub.schemas import SeparatedAudio, SynthesizedAudio, Transcript


class PipelineContext(BaseModel):
    """Inputs and intermediate artifacts for a single recipe run."""

    # Set by the runner before the run starts.
    input_path: Path
    output_path: Path
    work_dir: Path
    settings: Settings

    # Filled in by stages as the recipe runs; each starts empty.
    audio: Path | None = None
    separated: SeparatedAudio | None = None
    transcript: Transcript | None = None
    translation: Transcript | None = None
    synthesized: SynthesizedAudio | None = None
    fitted_vocals: Path | None = None
    mixed_audio: Path | None = None
    subtitle_path: Path | None = None

    def path(self, name: str) -> Path:
        """An intermediate-file path inside this run's work directory."""
        return self.work_dir / name
