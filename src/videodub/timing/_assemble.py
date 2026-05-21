"""Shared timeline assembly for the timing backends.

Both timing backends finish the same way: a set of speech clips, each with an
absolute start time, laid onto one continuous mono track with silence filling
the gaps. Only the clips differ — the mock places them untouched, the rubberband
fitter time-stretches them first — so the placement itself lives here, used by
both. Keeping it shared also keeps the two backends' output format identical.

Not part of the public API; internal to the `timing` package.
"""

from __future__ import annotations

from pathlib import Path

from videodub._ffmpeg import make_silence, run_ffmpeg


def place_on_timeline(
    clips: list[tuple[Path, float]], sample_rate: int, out: Path
) -> Path:
    """Lay each `(clip, start_seconds)` onto one mono track written to `out`.

    Every clip is delayed to its start time and the delayed streams are summed,
    so the gaps between clips stay silent and the track keeps the same timeline
    as the original video. With no clips the result is an empty silent track.
    Returns `out`.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # No clips -> an empty (zero-length) silent track. Nothing to place.
    if not clips:
        return make_silence(out, duration=0.0, sample_rate=sample_rate)

    # One ffmpeg input per clip; `adelay` shifts clip i to its start time.
    inputs: list[str] = []
    delays: list[tuple[int, int]] = []  # (input index, delay in ms)
    for index, (path, start) in enumerate(clips):
        inputs += ["-i", str(path)]
        delays.append((index, max(round(start * 1000), 0)))

    if len(delays) == 1:
        # `amix` needs at least two inputs; a single delayed clip *is* the whole
        # track, so delay it straight into the output label.
        index, delay_ms = delays[0]
        filtergraph = f"[{index}:a]adelay={delay_ms}:all=1[out]"
    else:
        # Delay each clip, then sum the delayed streams. Summing lays the clips
        # onto the timeline at their start times; the gaps stay silent. `amix`
        # runs until its longest input ends, so the track lasts until the final
        # clip finishes.
        parts: list[str] = []
        labels: list[str] = []
        for index, delay_ms in delays:
            label = f"d{index}"
            parts.append(f"[{index}:a]adelay={delay_ms}:all=1[{label}]")
            labels.append(f"[{label}]")
        parts.append(
            "".join(labels)
            + f"amix=inputs={len(labels)}:normalize=0:dropout_transition=0[out]"
        )
        filtergraph = ";".join(parts)

    run_ffmpeg(
        [
            *inputs,
            "-filter_complex", filtergraph,
            "-map", "[out]",
            "-ac", "1",
            "-ar", str(sample_rate),
            str(out),
        ]
    )
    return out
