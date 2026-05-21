"""Parse subtitle text back into a Transcript — the inverse of `render`. PORTABLE.

`render` turns a Transcript into subtitle text; `parse` reads it back. This is
the entry point for the `refine_subtitles` recipe, which loads an existing
subtitle file, proofreads it, and writes it out again.

Only SRT and VTT are parsed. Both are the same shape — blank-line-separated
cues, each a `START --> END` line plus text — and differ only in punctuation
(`,` vs `.` before the milliseconds) and a `WEBVTT` header, so one code path
reads both. ASS, with its INI-style sections, is render-only.

Pure text: no GPU, no network, no system binary.
"""

from __future__ import annotations

import re
from pathlib import Path

from videodub.schemas import Segment, Transcript

# A subtitle timestamp, accepting either spelling: SRT `00:00:01,500` and VTT
# `00:00:01.500`, plus VTT's hour-less `01:02.500` short form.
_TIMESTAMP = re.compile(r"(?:(\d+):)?([0-5]\d):([0-5]\d)[.,](\d{3})")
_ARROW = "-->"  # separates the start and end timestamps on a cue's timing line


def _to_seconds(token: str) -> float | None:
    """Parse one timestamp to seconds, or `None` if `token` is not a timestamp."""
    match = _TIMESTAMP.fullmatch(token.strip())
    if match is None:
        return None
    hours = int(match.group(1)) if match.group(1) else 0
    minutes, seconds, millis = (int(match.group(i)) for i in (2, 3, 4))
    return hours * 3600 + minutes * 60 + seconds + millis / 1000


def _parse_timing(line: str) -> tuple[float, float]:
    """Read `(start, end)` in seconds from a cue's `START --> END` line.

    A VTT timing line may carry cue-position settings after the end time
    (`... --> 00:00:04.000 line:0`); only the leading timestamp is read.

    Raises `ValueError` if either timestamp will not parse.
    """
    before, _, after = line.partition(_ARROW)
    after_tokens = after.split()
    start = _to_seconds(before)
    end = _to_seconds(after_tokens[0]) if after_tokens else None
    if start is None or end is None:
        raise ValueError(
            f"subtitle cue has an unparseable timestamp line: {line!r}"
        )
    return start, end


def parse(text: str, language: str | None = None) -> Transcript:
    """Parse SRT or VTT `text` into a Transcript.

    Cues are separated by blank lines; each is an optional index line, a
    `START --> END` line, then one or more text lines. A `WEBVTT` header and
    any block with no timing line (e.g. a VTT `NOTE`) are skipped. `language`
    is recorded on the result as given — subtitle files do not state their own.

    Raises `ValueError` if a cue's timing line cannot be parsed.
    """
    segments: list[Segment] = []
    # Split on one-or-more blank lines, tolerating CRLF and trailing spaces.
    for block in re.split(r"(?:\r?\n[ \t]*){2,}", text.strip()):
        lines = block.splitlines()
        if not lines or lines[0].strip().upper().startswith("WEBVTT"):
            continue  # an empty block, or the VTT file header — not a cue
        arrow_index = next(
            (i for i, line in enumerate(lines) if _ARROW in line), None
        )
        if arrow_index is None:
            continue  # a VTT NOTE block or stray text — nothing timed here
        start, end = _parse_timing(lines[arrow_index])
        # Lines before the arrow are the cue index/id; lines after are the text.
        cue_text = "\n".join(lines[arrow_index + 1 :]).strip()
        segments.append(Segment(start=start, end=end, text=cue_text))
    return Transcript(segments=segments, language=language)


def load(path: Path | str, language: str | None = None) -> Transcript:
    """Read a subtitle file from disk and parse it into a Transcript.

    The file is read as UTF-8, so non-Latin scripts survive. SRT and VTT are
    supported; ASS is render-only and rejected here.

    Raises `ValueError` for an ASS file or unparseable content.
    """
    path = Path(path)
    if path.suffix.lower() == ".ass":
        raise ValueError(
            "parsing ASS subtitles is not supported; pass an SRT or VTT file"
        )
    return parse(path.read_text(encoding="utf-8"), language=language)
