"""The translation stage's backend contract.

A `Translator` turns a `Transcript` into a *new* `Transcript`: the same number
of segments, the same timestamps, translated `text`. That 1:1, timestamp-
preserving guarantee is what lets downstream stages align source and translated
segments by list index — so `build_translation` enforces it in one place and
every backend funnels its output through it.

This module imports only `schemas`, `config`, and `errors` — no httpx, no
network — so it stays cheap to import.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from videodub.config import TranslationConfig
from videodub.errors import BackendError
from videodub.schemas import Segment, Transcript


class Translator(ABC):
    """A swappable translation backend, selected by `TranslationConfig.backend`."""

    @abstractmethod
    def translate(self, transcript: Transcript, cfg: TranslationConfig) -> Transcript:
        """Translate every segment, returning a new 1:1 timestamp-aligned Transcript."""

    def refine(self, transcript: Transcript, cfg: TranslationConfig) -> Transcript:
        """Proofread the *source-language* transcript, fixing ASR mistakes.

        The transcript came from a speech-recognition model, so it can carry
        transcription errors. A backend that can proofread returns a new
        transcript in the **same language** with corrected text and unchanged
        timestamps; lines that were already correct come back untouched.

        The default is a pass-through — backends that cannot proofread (the
        mock, and any not-yet-built backend) return the transcript as given, so
        the refine pipeline stage is harmless for them. `DeepSeekTranslator`
        overrides it with a real correction pass.
        """
        return transcript


def build_translation(
    source: Transcript, texts: list[str], cfg: TranslationConfig
) -> Transcript:
    """Assemble the translated `Transcript` from `source` and per-segment `texts`.

    `texts[i]` becomes the text of segment `i`; start, end, and speaker are
    carried over from the source untouched. The result's `language` is the
    target language and `source_language` records the original.

    Raises `BackendError` unless `texts` has exactly one entry per source
    segment — the contract every backend must honour.
    """
    if len(texts) != len(source.segments):
        raise BackendError(
            f"translation is not 1:1: got {len(texts)} text(s) for "
            f"{len(source.segments)} segment(s)"
        )
    segments = [
        Segment(start=src.start, end=src.end, text=text, speaker=src.speaker)
        for src, text in zip(source.segments, texts, strict=True)
    ]
    return Transcript(
        segments=segments,
        language=cfg.target_language,
        source_language=cfg.source_language,
    )


def build_refinement(source: Transcript, texts: list[str]) -> Transcript:
    """Assemble the proofread transcript from `source` and corrected `texts`.

    The mirror of `build_translation` for the correction pass: `texts[i]`
    replaces the text of segment `i`, while start, end, speaker, and — crucially
    — the **language** are carried over from `source` untouched. Refinement
    fixes the source text, it does not translate it.

    Raises `BackendError` unless `texts` has exactly one entry per source
    segment.
    """
    if len(texts) != len(source.segments):
        raise BackendError(
            f"refinement is not 1:1: got {len(texts)} text(s) for "
            f"{len(source.segments)} segment(s)"
        )
    segments = [
        Segment(start=src.start, end=src.end, text=text, speaker=src.speaker)
        for src, text in zip(source.segments, texts, strict=True)
    ]
    return Transcript(
        segments=segments,
        language=source.language,
        source_language=source.source_language,
    )
