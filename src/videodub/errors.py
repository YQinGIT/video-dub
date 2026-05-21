"""Shared exception types.

Like `schemas`, this module depends on nothing else in the package.
"""


class VideodubError(Exception):
    """Base class for all videodub errors."""


class ConfigError(VideodubError):
    """Invalid or missing configuration."""


class BackendError(VideodubError):
    """A backend (ASR / translation / TTS / ...) failed or is unavailable."""


class StageError(VideodubError):
    """A pipeline stage failed during execution."""


class MediaIOError(VideodubError):
    """An ffmpeg / ffprobe call failed, or the binary is missing from PATH."""
