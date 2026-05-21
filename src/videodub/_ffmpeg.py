"""Private ffmpeg helper for the stages that synthesize or combine audio.

The mock `separation`, `tts`, and `timing` backends fabricate audio — silent
tracks, sine tones, clips laid onto a timeline — and the `mixing` stage layers
the dubbed vocals over the background. Each of them shells out to ffmpeg — and,
to read a clip's length, ffprobe — and this module is the single place doing so.

Not part of the public API. `media_io` has its own subprocess wrapper; the two
stay separate on purpose, so that no stage has to import another.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from videodub.errors import BackendError


def run_ffmpeg(args: list[str]) -> None:
    """Run ffmpeg with `args` — everything that follows the `ffmpeg` binary.

    `-nostdin` (never block waiting for keyboard input) and `-y` (overwrite the
    output without prompting) are always prepended.

    Raises `BackendError` if ffmpeg is missing from PATH or exits non-zero.
    """
    argv = ["ffmpeg", "-nostdin", "-y", *args]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise BackendError(
            "`ffmpeg` not found on PATH — it is required by the mock audio "
            "backends; install ffmpeg and try again."
        ) from exc

    if proc.returncode != 0:
        raise BackendError(
            f"ffmpeg failed (exit code {proc.returncode}):\n{proc.stderr.strip()}"
        )


def make_silence(out: Path, duration: float, sample_rate: int) -> Path:
    """Write `duration` seconds of mono silence to `out`, returning `out`.

    A negative `duration` is clamped to zero.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-f", "lavfi",
            "-i", f"anullsrc=sample_rate={sample_rate}:channel_layout=mono",
            "-t", f"{max(duration, 0.0):.6f}",
            str(out),
        ]
    )
    return out


def make_tone(
    out: Path, duration: float, sample_rate: int, frequency: float = 220.0
) -> Path:
    """Write `duration` seconds of a mono `frequency`-Hz sine tone to `out`.

    A negative `duration` is clamped to zero. Returns `out`.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-f", "lavfi",
            "-i", (
                f"sine=frequency={frequency}"
                f":sample_rate={sample_rate}"
                f":duration={max(duration, 0.0):.6f}"
            ),
            "-ac", "1",
            str(out),
        ]
    )
    return out


def audio_duration(path: Path) -> float:
    """Return the duration, in seconds, of the audio file at `path`.

    Shells out to `ffprobe` (which ships alongside ffmpeg). Raises `BackendError`
    if ffprobe is missing, exits non-zero, or reports no parseable duration.
    """
    argv = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise BackendError(
            "`ffprobe` not found on PATH — it ships with ffmpeg and is needed to "
            "measure clip lengths; install ffmpeg and try again."
        ) from exc

    if proc.returncode != 0:
        raise BackendError(
            f"ffprobe failed (exit code {proc.returncode}):\n{proc.stderr.strip()}"
        )

    try:
        return float(proc.stdout.strip())
    except ValueError as exc:
        raise BackendError(
            f"ffprobe returned no usable duration for {path}: {proc.stdout!r}"
        ) from exc
