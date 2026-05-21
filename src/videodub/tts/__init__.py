"""tts — translated Transcript to cloned speech. CUDA-BOUND (CosyVoice 2).

Public API:
    get_tts_backend(cfg) -> TTSBackend   select a backend by config

`get_tts_backend` imports the chosen backend lazily, so importing this package
never pulls in a heavy voice model unless a real backend is actually used.
"""

from __future__ import annotations

from videodub.config import TTSConfig
from videodub.errors import ConfigError
from videodub.tts.base import TTSBackend

__all__ = ["TTSBackend", "get_tts_backend"]

# Real backends are deferred to Stage 7c. CosyVoice 2 is the planned default;
# GPT-SoVITS and ElevenLabs are alternates kept in the config for later.
_DEFERRED = {"cosyvoice2", "gpt_sovits", "elevenlabs"}


def get_tts_backend(cfg: TTSConfig) -> TTSBackend:
    """Return the TTS backend named by `cfg.backend`.

    Raises `ConfigError` if the backend is unknown or not yet implemented.
    """
    if cfg.backend == "mock":
        from videodub.tts.mock import MockTTS

        return MockTTS()

    if cfg.backend in _DEFERRED:
        raise ConfigError(
            f"TTS backend {cfg.backend!r} is not implemented yet "
            "(planned for Stage 7c); use 'mock' for now."
        )

    raise ConfigError(f"unknown TTS backend: {cfg.backend!r}")
