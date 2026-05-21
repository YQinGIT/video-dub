"""Private helper: run an external command and turn failure into MediaIOError.

Not part of the public API — everything here is for use inside `media_io` only.
"""

from __future__ import annotations

import subprocess

from videodub.errors import MediaIOError


def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Run `argv`, returning the completed process on success.

    `argv` is the command as a list, e.g. ``["ffmpeg", "-i", "in.mp4", ...]``.
    Passing a list (not a string) means no shell is involved, so spaces or odd
    characters in file paths can never be misread as extra arguments.

    Raises `MediaIOError` if the binary is missing from PATH, or if the command
    runs but exits non-zero. ffmpeg/ffprobe write their diagnostics to stderr,
    so that text is forwarded into the exception message.
    """
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,  # collect stdout/stderr instead of printing them
            text=True,  # decode bytes to str using the default encoding
            check=False,  # we inspect returncode ourselves for a better message
        )
    except FileNotFoundError as exc:
        raise MediaIOError(
            f"`{argv[0]}` not found on PATH — install ffmpeg and try again."
        ) from exc

    if proc.returncode != 0:
        raise MediaIOError(
            f"`{argv[0]}` failed (exit code {proc.returncode}):\n"
            f"{proc.stderr.strip()}"
        )
    return proc
