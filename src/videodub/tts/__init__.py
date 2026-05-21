"""tts — translated Transcript to cloned speech. CUDA-BOUND.

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

# GPT-SoVITS and ElevenLabs are kept in the config as documented alternates but
# are not implemented; IndexTTS-2 is the Stage 7c TTS backend.
_DEFERRED = {"gpt_sovits", "elevenlabs"}


def get_tts_backend(cfg: TTSConfig) -> TTSBackend:
    """Return the TTS backend named by `cfg.backend`.

    Raises `ConfigError` if the backend is unknown or not implemented.
    """
    if cfg.backend == "mock":
        from videodub.tts.mock import MockTTS

        return MockTTS()

    if cfg.backend == "indextts2":
        # The IndexTTS-2 backend imports cheaply — it drives IndexTTS-2 out of
        # process, so it pulls in no heavy library here. A missing install is
        # reported by `synthesize()` at run time, not by this import.
        from videodub.tts.indextts2 import IndexTTS2TTS

        return IndexTTS2TTS()

    if cfg.backend in _DEFERRED:
        raise ConfigError(
            f"TTS backend {cfg.backend!r} is not implemented; "
            "use 'indextts2' or 'mock'."
        )

    raise ConfigError(f"unknown TTS backend: {cfg.backend!r}")
