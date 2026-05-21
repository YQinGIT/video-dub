"""Private ffmpeg helper for the mock GPU backends.

The mock `separation`, `tts`, and `timing` backends fabricate audio — silent
tracks, sine tones, clips laid onto a timeline — so the whole pipeline runs end
to end with no CUDA. Each of them shells out to ffmpeg, and this module is the
single place that does so.

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
