"""The recipe catalogue — pure data, no logic.

A recipe is an ordered list of stage names. The runner walks the list; the
stage functions themselves live in `stages.py`. Adding a recipe means adding an
entry here and nothing else — recipes carry no recipe-specific code.

`separation` appears in `full_dub` but is a *toggle*, not a separate recipe: the
runner skips it when `SeparationConfig.enabled` is False.
"""

from __future__ import annotations

RECIPES: dict[str, list[str]] = {
    "full_dub": [
        "extract_audio", "separation", "asr", "translation",
        "tts", "timing", "mixing", "remux",
    ],
    "translate_subtitles": ["extract_audio", "asr", "translation", "subtitle"],
    "transcribe": ["extract_audio", "asr", "subtitle"],
}
