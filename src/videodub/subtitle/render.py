"""Render a Transcript to subtitle text — SRT, VTT, or ASS. PORTABLE.

A subtitle file is just timed text: "show this string from time A to time B".
The three supported formats carry the same information and differ only in their
framing and in how they spell a timestamp:

    SRT   00:00:01,500   numbered cues, a comma before the milliseconds
    VTT   00:00:01.500   a `WEBVTT` header, a period before the milliseconds
    ASS   0:00:01.50     INI-style sections, centiseconds, single-digit hour

`render` returns the file *content* as a string; `write` saves it to disk.
Passing `secondary` stacks a second language under each cue — bilingual output.

This module is pure text formatting: no GPU, no network, no system binary.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from videodub.schemas import Segment, Transcript

SubtitleFormat = Literal["srt", "vtt", "ass"]

# Used by `write` to pick a format when the caller does not pass one explicitly.
_EXT_TO_FMT: dict[str, SubtitleFormat] = {".srt": "srt", ".vtt": "vtt", ".ass": "ass"}


# --------------------------------------------------------------------------- #
# Timestamp formatting — one helper per format, since each spells time its way #
# --------------------------------------------------------------------------- #

def _hmsms(seconds: float) -> tuple[int, int, int, int]:
    """Split a duration in seconds into (hours, minutes, seconds, milliseconds).

    Rounds to the nearest millisecond — subtitle formats have no finer unit.
    """
    total_ms = round(seconds * 1000)
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return h, m, s, ms


def _ts_srt(seconds: float) -> str:
    """`HH:MM:SS,mmm` — the SRT timestamp (comma before the milliseconds)."""
    h, m, s, ms = _hmsms(seconds)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ts_vtt(seconds: float) -> str:
    """`HH:MM:SS.mmm` — the VTT timestamp (period before the milliseconds)."""
    h, m, s, ms = _hmsms(seconds)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _ts_ass(seconds: float) -> str:
    """`H:MM:SS.cc` — the ASS timestamp (centiseconds, single-digit hour)."""
    total_cs = round(seconds * 100)
    h, rem = divmod(total_cs, 360_000)
    m, rem = divmod(rem, 6_000)
    s, cs = divmod(rem, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


# --------------------------------------------------------------------------- #
# Cue assembly                                                                #
# --------------------------------------------------------------------------- #

def _pairs(
    transcript: Transcript, secondary: Transcript | None
) -> Iterator[tuple[Segment, str | None]]:
    """Yield `(segment, secondary_text)` for every cue.

    `secondary_text` is `None` for single-language output. For bilingual output
    it is the matching segment's text from `secondary`; translation is a 1:1,
    timestamp-preserving operation, so the two transcripts must have the same
    number of segments.
    """
    if secondary is None:
        for seg in transcript.segments:
            yield seg, None
        return

    if len(secondary.segments) != len(transcript.segments):
        raise ValueError(
            f"bilingual rendering needs a 1:1 segment match: primary has "
            f"{len(transcript.segments)} segment(s), secondary has "
            f"{len(secondary.segments)}"
        )
    for seg, sec in zip(transcript.segments, secondary.segments, strict=True):
        yield seg, sec.text


def _cue_text(seg: Segment, sec_text: str | None) -> str:
    """The displayed text for one cue — primary alone, or primary over secondary.

    SRT and VTT both take a real newline as an in-cue line break.
    """
    if sec_text is None:
        return seg.text
    return f"{seg.text}\n{sec_text}"


# --------------------------------------------------------------------------- #
# Per-format renderers                                                        #
# --------------------------------------------------------------------------- #

def _render_srt(transcript: Transcript, secondary: Transcript | None) -> str:
    """SRT: blank-line-separated cues, each numbered from 1."""
    blocks: list[str] = []
    for i, (seg, sec_text) in enumerate(_pairs(transcript, secondary), start=1):
        blocks.append(
            "\n".join(
                [
                    str(i),
                    f"{_ts_srt(seg.start)} --> {_ts_srt(seg.end)}",
                    _cue_text(seg, sec_text),
                ]
            )
        )
    # A trailing newline is conventional; an empty transcript yields an empty file.
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _render_vtt(transcript: Transcript, secondary: Transcript | None) -> str:
    """VTT: a `WEBVTT` header, then blank-line-separated, unnumbered cues."""
    blocks: list[str] = ["WEBVTT"]
    for seg, sec_text in _pairs(transcript, secondary):
        blocks.append(
            "\n".join(
                [
                    f"{_ts_vtt(seg.start)} --> {_ts_vtt(seg.end)}",
                    _cue_text(seg, sec_text),
                ]
            )
        )
    return "\n\n".join(blocks) + "\n"


# ASS keeps its styling in a fixed preamble. Defined as field lists joined at
# import time so no source line runs long; the runtime strings are single lines.
_ASS_STYLE_FORMAT = (
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
    "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
    "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
    "MarginL, MarginR, MarginV, Encoding"
)
_ASS_STYLE = (
    "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
    "0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1"
)
_ASS_EVENT_FORMAT = (
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
    "Effect, Text"
)


def _render_ass(transcript: Transcript, secondary: Transcript | None) -> str:
    """ASS: `[Script Info]` / `[V4+ Styles]` / `[Events]` sections.

    ASS has no real newline inside a line — an in-cue break is the literal
    two-character escape ``\\N``, so a bilingual cue joins its languages with it.
    """
    lines: list[str] = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "",
        "[V4+ Styles]",
        _ASS_STYLE_FORMAT,
        _ASS_STYLE,
        "",
        "[Events]",
        _ASS_EVENT_FORMAT,
    ]
    for seg, sec_text in _pairs(transcript, secondary):
        parts = [seg.text] if sec_text is None else [seg.text, sec_text]
        text = "\n".join(parts).replace("\n", r"\N")
        lines.append(
            f"Dialogue: 0,{_ts_ass(seg.start)},{_ts_ass(seg.end)},"
            f"Default,,0,0,0,,{text}"
        )
    return "\n".join(lines) + "\n"


_RENDERERS = {"srt": _render_srt, "vtt": _render_vtt, "ass": _render_ass}


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #

def render(
    transcript: Transcript,
    fmt: SubtitleFormat,
    secondary: Transcript | None = None,
) -> str:
    """Render `transcript` to subtitle file content in `fmt` ('srt'/'vtt'/'ass').

    Pass `secondary` to stack a second language under each cue (bilingual
    output); it must have the same number of segments as `transcript`.

    Raises `ValueError` for an unknown `fmt` or a mismatched `secondary`.
    """
    try:
        renderer = _RENDERERS[fmt]
    except KeyError:
        raise ValueError(
            f"unknown subtitle format {fmt!r}; choose from {sorted(_RENDERERS)}"
        ) from None
    return renderer(transcript, secondary)


def write(
    transcript: Transcript,
    path: Path | str,
    fmt: SubtitleFormat | None = None,
    secondary: Transcript | None = None,
) -> Path:
    """Render `transcript` and write it to `path`, returning `path`.

    When `fmt` is omitted it is inferred from the file extension (`.srt`,
    `.vtt`, `.ass`). Missing parent directories are created. The file is written
    as UTF-8 so non-Latin scripts (e.g. Chinese) survive the round trip.

    Raises `ValueError` if the format cannot be determined or is unknown.
    """
    path = Path(path)
    if fmt is None:
        fmt = _EXT_TO_FMT.get(path.suffix.lower())
        if fmt is None:
            raise ValueError(
                f"cannot infer subtitle format from extension {path.suffix!r}; "
                f"pass fmt= explicitly or use one of {sorted(_EXT_TO_FMT)}"
            )
    content = render(transcript, fmt, secondary=secondary)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
