"""separation — vocal / background split. CUDA-BOUND (Demucs).

Public API:
    get_separator(cfg) -> Separator   select a backend by config

`get_separator` imports the chosen backend lazily, so importing this package
never pulls in `torch` unless the real Demucs backend is actually used.
"""

from __future__ import annotations

from videodub.config import SeparationConfig
from videodub.errors import ConfigError
from videodub.separation.base import Separator

__all__ = ["Separator", "get_separator"]


def get_separator(cfg: SeparationConfig) -> Separator:
    """Return the separation backend named by `cfg.backend`.

    Raises `ConfigError` if the backend is unknown or not yet implemented.
    """
    if cfg.backend == "mock":
        from videodub.separation.mock import MockSeparator

        return MockSeparator()

    if cfg.backend == "demucs":
        raise ConfigError(
            "separation backend 'demucs' is not implemented yet "
            "(planned for Stage 7b); use 'mock' for now."
        )

    raise ConfigError(f"unknown separation backend: {cfg.backend!r}")
