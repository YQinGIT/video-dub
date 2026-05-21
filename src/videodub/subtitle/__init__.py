"""subtitle — Transcript to SRT/VTT/ASS. PORTABLE.

Public API:
    render(transcript, fmt, secondary=None) -> str      subtitle file content
    write(transcript, path, fmt=None, secondary=None) -> Path   render and save

`fmt` is one of "srt", "vtt", "ass". Pass `secondary` to stack a second
language under each cue (bilingual subtitles).
"""

from videodub.subtitle.render import SubtitleFormat, render, write

__all__ = ["SubtitleFormat", "render", "write"]
