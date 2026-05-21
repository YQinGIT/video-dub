"""timing — fit synth segments to target durations. PORTABLE (rubberband binary).

Public API:
    get_timing_fitter(cfg) -> TimingFitter   select a backend by config

`get_timing_fitter` imports the chosen backend lazily, keeping this package
cheap to import.
"""

from __future__ import annotations

from videodub.config import TimingConfig
from videodub.errors import ConfigError
from videodub.timing.base import TimingFitter

__all__ = ["TimingFitter", "get_timing_fitter"]


def get_timing_fitter(cfg: TimingConfig) -> TimingFitter:
    """Return the timing-fit backend named by `cfg.backend`.

    Raises `ConfigError` if the backend is unknown or not yet implemented.
    """
    if cfg.backend == "mock":
        from videodub.timing.mock import MockTimingFitter

        return MockTimingFitter()

    if cfg.backend == "rubberband":
        raise ConfigError(
            "timing backend 'rubberband' is not implemented yet "
            "(planned for Stage 7d); use 'mock' for now."
        )

    raise ConfigError(f"unknown timing backend: {cfg.backend!r}")
