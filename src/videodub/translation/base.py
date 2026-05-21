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
