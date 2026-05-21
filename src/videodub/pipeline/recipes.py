"""The recipe catalogue — pure data, no logic.

A recipe is an ordered list of stage names. The runner walks the list; the
stage functions themselves live in `stages.py`. Adding a recipe means adding an
entry here and nothing else — recipes carry no recipe-specific code.

`separation` and `refine` appear inside other recipes as *toggles*, not as
recipes of their own: the runner skips `separation` when `SeparationConfig.
enabled` is False and `refine` when `TranslationConfig.refine_source` is False.

`refine_subtitles` is the standalone correction recipe: it takes an existing
subtitle file (not a video), proofreads the ASR text with DeepSeek, and writes
the corrected file back. `transcribe` stays a pure offline speech-to-text path
— it never calls the translation API.
"""

from __future__ import annotations

RECIPES: dict[str, list[str]] = {
    "full_dub": [
        "extract_audio", "separation", "asr", "refine", "translation",
        "tts", "timing", "mixing", "remux",
    ],
    "translate_subtitles": [
        "extract_audio", "asr", "refine", "translation", "subtitle",
    ],
    "transcribe": ["extract_audio", "asr", "subtitle"],
    "refine_subtitles": ["load_subtitle", "refine", "subtitle"],
}
