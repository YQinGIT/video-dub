"""subtitle — Transcript <-> SRT/VTT/ASS. PORTABLE.

Public API:
    render(transcript, fmt, secondary=None) -> str      subtitle file content
    write(transcript, path, fmt=None, secondary=None) -> Path   render and save
    parse(text, language=None) -> Transcript            SRT/VTT text -> Transcript
    load(path, language=None) -> Transcript             read a file and parse it

`fmt` is one of "srt", "vtt", "ass". Pass `secondary` to stack a second
language under each cue (bilingual subtitles). `parse` and `load` are the
inverse of `render` — they read SRT or VTT back into a Transcript; ASS is
render-only.
"""

from videodub.subtitle.parse import load, parse
from videodub.subtitle.render import SubtitleFormat, render, write

__all__ = ["SubtitleFormat", "load", "parse", "render", "write"]
