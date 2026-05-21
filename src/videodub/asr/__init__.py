"""asr — audio to Transcript. CUDA-BOUND (faster-whisper).

Public API:
    get_asr_backend(cfg) -> ASRBackend   select a backend by config

`get_asr_backend` imports the chosen backend lazily, so importing this package
never pulls in `torch` unless a real CUDA backend is actually used.
"""

from __future__ import annotations

from videodub.asr.base import ASRBackend
from videodub.config import ASRConfig
from videodub.errors import ConfigError

__all__ = ["ASRBackend", "get_asr_backend"]


def get_asr_backend(cfg: ASRConfig) -> ASRBackend:
    """Return the ASR backend named by `cfg.backend`.

    Raises `ConfigError` if the backend is unknown or not yet implemented.
    """
    if cfg.backend == "mock":
        from videodub.asr.mock import MockASR

        return MockASR()

    if cfg.backend == "faster_whisper":
        from videodub.asr.faster_whisper import FasterWhisperASR

        return FasterWhisperASR()

    if cfg.backend == "whisperx":
        from videodub.asr.whisperx import WhisperXASR

        return WhisperXASR()

    if cfg.backend == "funasr":
        from videodub.asr.funasr import FunASRASR

        return FunASRASR()

    raise ConfigError(f"unknown ASR backend: {cfg.backend!r}")
