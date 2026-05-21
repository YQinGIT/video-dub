"""Deterministic, offline translation backend.

`MockTranslator` does not really translate — it tags each segment's text with
the target language code (e.g. "你好" -> "[en] 你好"). It needs no network and no
API key, so the whole pipeline can run and be tested on any machine. Its output
is fully deterministic, which lets pipeline tests assert on exact text.
"""

from __future__ import annotations

from videodub.config import TranslationConfig
from videodub.schemas import Transcript
from videodub.translation.base import Translator, build_translation


class MockTranslator(Translator):
    """A fake translator: prefixes every segment with the target-language tag."""

    def translate(self, transcript: Transcript, cfg: TranslationConfig) -> Transcript:
        texts = [f"[{cfg.target_language}] {seg.text}" for seg in transcript.segments]
        return build_translation(transcript, texts, cfg)
