"""Transform media with ffmpeg: pull audio out of a video, put audio back in."""

from __future__ import annotations

from pathlib import Path

from videodub.errors import MediaIOError
from videodub.media_io._subprocess import run


def _require_file(path: Path, label: str) -> Path:
    """Return `path` as a Path, raising MediaIOError if it does not exist."""
    path = Path(path)
    if not path.exists():
        raise MediaIOError(f"{label} not found: {path}")
    return path


def _prepare_output(out: Path | str) -> Path:
    """Coerce `out` to a Path and make sure its parent directory exists."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def extract_audio(
    video: Path | str,
    out: Path | str,
    sr: int = 16000,
    mono: bool = True,
) -> Path:
    """Extract the audio track of `video` into `out`, returning `out`.

    `sr` sets the output sample rate in Hz (16 kHz is what ASR models expect).
    `mono` mixes every channel down to one. The output container is chosen by
    `out`'s extension — use `.wav` for lossless PCM.

    Raises `MediaIOError` if `video` is missing or ffmpeg fails.
    """
    video = _require_file(video, "input video")
    out = _prepare_output(out)

    argv = [
        "ffmpeg",
        "-y",  # overwrite the output file without prompting
        "-i", str(video),
        "-vn",  # drop the video stream — audio only
        "-ar", str(sr),  # resample
    ]
    if mono:
        argv += ["-ac", "1"]  # downmix to a single channel
    argv.append(str(out))

    run(argv)
    return out


def remux(video: Path | str, audio: Path | str, out: Path | str) -> Path:
    """Combine the picture of `video` with the audio of `audio` into `out`.

    The video stream is copied as-is (fast, lossless); the audio is encoded to
    AAC so the result plays in standard `.mp4` players. `-shortest` ends the
    output when the shorter of the two inputs runs out.

    Raises `MediaIOError` if an input is missing or ffmpeg fails.
    """
    video = _require_file(video, "input video")
    audio = _require_file(audio, "input audio")
    out = _prepare_output(out)

    argv = [
        "ffmpeg",
        "-y",
        "-i", str(video),  # input 0
        "-i", str(audio),  # input 1
        "-map", "0:v:0",  # take the first video stream from input 0
        "-map", "1:a:0",  # take the first audio stream from input 1
        "-c:v", "copy",  # keep the video bytes untouched — no re-encode
        "-c:a", "aac",  # encode the new audio to AAC
        "-shortest",
        str(out),
    ]
    run(argv)
    return out
