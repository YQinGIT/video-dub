"""Mix the dubbed vocals with the preserved background into one audio track.

`mixing` is the last audio stage of a dub: the synthesized, timed vocal track
goes on top, the original music-and-effects background sits underneath, each
with its own gain. It is portable — ffmpeg only, no GPU.
"""

from __future__ import annotations

from pathlib import Path

from videodub._ffmpeg import run_ffmpeg
from videodub.config import MixingConfig
from videodub.errors import BackendError

# Both inputs are resampled to this rate before mixing: the dubbed vocals and
# the background generally arrive at different sample rates, and ffmpeg's
# `amix` filter requires its inputs to agree. 48 kHz is the video standard.
_MIX_SAMPLE_RATE = 48000


def mix(
    vocals: Path | str,
    background: Path | str,
    out: Path | str,
    cfg: MixingConfig,
) -> Path:
    """Layer `vocals` over `background`, applying each track's gain, into `out`.

    Gains come from `cfg` in decibels (0 dB leaves a track unchanged). The two
    inputs are resampled to a common rate and summed; the result runs as long
    as the longer input. Returns `out`.

    Raises `BackendError` if an input file is missing or ffmpeg fails.
    """
    vocals, background, out = Path(vocals), Path(background), Path(out)
    for label, path in (("vocals", vocals), ("background", background)):
        if not path.exists():
            raise BackendError(f"{label} audio not found: {path}")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Input 0 is the vocals, input 1 the background. Apply gain, resample both
    # to a common rate, then sum them. `normalize=0` keeps the gains as given
    # (amix would otherwise attenuate by the input count); `dropout_transition=0`
    # avoids a volume ramp when the shorter input ends.
    filtergraph = (
        f"[0:a]volume={cfg.vocal_gain_db}dB,aresample={_MIX_SAMPLE_RATE}[v];"
        f"[1:a]volume={cfg.background_gain_db}dB,aresample={_MIX_SAMPLE_RATE}[b];"
        "[v][b]amix=inputs=2:normalize=0:dropout_transition=0[out]"
    )
    run_ffmpeg(
        [
            "-i", str(vocals),
            "-i", str(background),
            "-filter_complex", filtergraph,
            "-map", "[out]",
            "-ac", "1",
            str(out),
        ]
    )
    return out
