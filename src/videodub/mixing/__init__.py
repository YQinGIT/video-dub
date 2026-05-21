"""mixing — dubbed vocals + preserved background -> final audio. PORTABLE (ffmpeg).

Public API:
    mix(vocals, background, out, cfg) -> Path   sum two tracks into one
"""

from videodub.mixing.mix import mix

__all__ = ["mix"]
