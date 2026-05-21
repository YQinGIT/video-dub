"""Inspect a media file with ffprobe and return a typed description of it."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from videodub.errors import MediaIOError
from videodub.media_io._subprocess import run


class StreamInfo(BaseModel):
    """One stream inside a media file — usually a video or an audio track.

    ffprobe emits dozens of fields per stream; we keep only the ones the
    pipeline cares about. Pydantic ignores the rest. Fields that apply to just
    one kind of stream (e.g. `width` for video) are optional.
    """

    index: int = Field(description="Position of this stream in the file")
    codec_type: str = Field(description="'video', 'audio', 'subtitle', ...")
    codec_name: str | None = Field(default=None, description="e.g. 'h264', 'aac'")
    duration: float | None = Field(default=None, description="Stream length, seconds")

    # Audio-only fields.
    sample_rate: int | None = Field(default=None, description="Samples per second (Hz)")
    channels: int | None = Field(default=None, description="Channel count (1 = mono)")

    # Video-only fields.
    width: int | None = Field(default=None, description="Frame width, pixels")
    height: int | None = Field(default=None, description="Frame height, pixels")


class MediaInfo(BaseModel):
    """The result of probing a file: its duration and every stream it holds."""

    path: Path = Field(description="The file that was probed")
    duration: float = Field(ge=0, description="Overall duration, seconds")
    streams: list[StreamInfo] = Field(default_factory=list)

    @property
    def audio_streams(self) -> list[StreamInfo]:
        """Every audio stream in the file."""
        return [s for s in self.streams if s.codec_type == "audio"]

    @property
    def video_streams(self) -> list[StreamInfo]:
        """Every video stream in the file."""
        return [s for s in self.streams if s.codec_type == "video"]

    @property
    def has_audio(self) -> bool:
        """True when the file carries at least one audio stream."""
        return bool(self.audio_streams)

    @property
    def has_video(self) -> bool:
        """True when the file carries at least one video stream."""
        return bool(self.video_streams)


def _safe_float(value: object) -> float | None:
    """Best-effort float parse. ffprobe sometimes reports 'N/A' or omits a field."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def probe(path: Path | str) -> MediaInfo:
    """Inspect `path` with ffprobe and return a `MediaInfo`.

    Raises `MediaIOError` if the file is missing or ffprobe cannot read it.
    """
    path = Path(path)
    if not path.exists():
        raise MediaIOError(f"file not found: {path}")

    proc = run(
        [
            "ffprobe",
            "-v", "error",  # silence everything except real errors
            "-print_format", "json",  # machine-readable output on stdout
            "-show_format",  # the container: duration, bitrate, ...
            "-show_streams",  # one entry per video/audio/subtitle track
            str(path),
        ]
    )
    data = json.loads(proc.stdout)

    streams = [StreamInfo.model_validate(s) for s in data.get("streams", [])]

    # Prefer the container's duration; fall back to the longest stream; else 0.
    duration = _safe_float(data.get("format", {}).get("duration"))
    if duration is None:
        stream_durations = [s.duration for s in streams if s.duration is not None]
        duration = max(stream_durations) if stream_durations else 0.0

    return MediaInfo(path=path, duration=duration, streams=streams)
