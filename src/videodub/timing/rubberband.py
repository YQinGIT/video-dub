"""Rubber Band timing-fit backend — Stage 7d.

The `tts` stage renders each translated line as its own clip, and a synthesized
line rarely lasts exactly as long as the segment it must fill: an English line
runs longer or shorter than the Chinese it was translated from. This backend
closes that gap. For every clip it measures the slot it must occupy against the
length it actually has, time-stretches the clip toward that slot, then lays
every stretched clip onto one continuous vocal track at its original timestamp —
so each line still lands where the original speaker's line was.

`rubberband` is a high-quality, pitch-preserving time-stretcher: it changes a
clip's *duration* without changing the speaker's pitch, so a stretched line
still sounds like the same voice rather than chipmunked or slowed to a drawl.

The stretch is clamped to `cfg.min_stretch`..`cfg.max_stretch`: past those
bounds speech stops sounding natural, so a clip needing more is left only
partly fitted — one too short keeps trailing silence (a slightly longer pause),
one too long is trimmed back to its slot so it cannot overrun the next line.
Trimming can clip a hair of speech off the end; a timing-aware translation
(`TranslationConfig.timing_aware`) is what keeps overshoots small enough that
this rarely bites.

`timing` is CPU-only: this backend shells out to the `rubberband` and `ffmpeg`
binaries and imports no GPU library. `rubberband` is a system package (no Python
bindings), called via subprocess the same way `ffmpeg` is.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from videodub._ffmpeg import audio_duration, run_ffmpeg
from videodub.config import TimingConfig
from videodub.errors import BackendError
from videodub.schemas import SynthesizedAudio
from videodub.timing._assemble import place_on_timeline
from videodub.timing.base import TimingFitter

# Below this much relative change, stretching is not worth doing: the clip
# already fits for practical purposes, and a needless trip through rubberband
# would only add processing artefacts. Such clips are placed unchanged. The
# same tolerance is the slack allowed before a clip counts as overrunning.
_STRETCH_EPSILON = 0.01


def _run_rubberband(time_ratio: float, src: Path, dst: Path) -> None:
    """Time-stretch `src` to `time_ratio`x its length, writing `dst`.

    `--time` is rubberband's duration multiplier: 1.3 makes the clip 1.3x longer
    (slower speech), 0.7 makes it shorter (faster); pitch is preserved either
    way. Raises `BackendError` if the binary is missing or exits non-zero.
    """
    argv = ["rubberband", "--time", f"{time_ratio:.6f}", str(src), str(dst)]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise BackendError(
            "`rubberband` not found on PATH — the rubberband timing backend "
            "needs the Rubber Band command-line tool; install it (e.g. "
            "`apt install rubberband-cli`) or use the 'mock' timing backend."
        ) from exc

    if proc.returncode != 0:
        raise BackendError(
            f"rubberband failed (exit code {proc.returncode}):\n{proc.stderr.strip()}"
        )


class RubberbandTimingFitter(TimingFitter):
    """Timing fit via the Rubber Band time-stretcher.

    Each clip is stretched (within the configured bounds) toward its segment's
    target duration, then all clips are laid onto one continuous track at their
    original start times.
    """

    def fit(self, synth: SynthesizedAudio, cfg: TimingConfig, out: Path) -> Path:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)

        # No clips -> an empty silent track; nothing to stretch or place.
        if not synth.segments:
            return place_on_timeline([], synth.sample_rate, out)

        if not 0 < cfg.min_stretch <= cfg.max_stretch:
            raise BackendError(
                f"invalid stretch bounds: min_stretch={cfg.min_stretch}, "
                f"max_stretch={cfg.max_stretch} (need 0 < min_stretch <= max_stretch)"
            )

        # Stretched / trimmed intermediates live beside the final track.
        work = out.parent / "_rubberband"
        work.mkdir(parents=True, exist_ok=True)

        clips: list[tuple[Path, float]] = []
        for index, seg in enumerate(synth.segments):
            src = Path(seg.audio_path)
            if not src.exists():
                raise BackendError(f"synth clip not found: {src}")

            target = seg.target_duration
            actual = audio_duration(src)

            # A zero-length clip or a zero-length slot has no meaningful ratio —
            # place the clip untouched and let the assembler deal with it.
            if actual <= 0 or target <= 0:
                clips.append((src, seg.start))
                continue

            # Stretch toward the slot, but only within the natural-sounding
            # bounds; `--time` past them would chipmunk or drawl the voice.
            ratio = min(max(target / actual, cfg.min_stretch), cfg.max_stretch)
            if abs(ratio - 1.0) < _STRETCH_EPSILON:
                fitted, fitted_len = src, actual
            else:
                fitted = work / f"stretch_{index:04d}.wav"
                _run_rubberband(ratio, src, fitted)
                fitted_len = actual * ratio

            # A clamped stretch can leave a clip still longer than its slot;
            # trim it so it cannot bleed into the next line. A clip shorter than
            # its slot is left as-is — the silent gap to the next clip reads as
            # a natural pause.
            if fitted_len > target + _STRETCH_EPSILON:
                trimmed = work / f"trim_{index:04d}.wav"
                run_ffmpeg(["-i", str(fitted), "-t", f"{target:.6f}", str(trimmed)])
                fitted = trimmed

            clips.append((fitted, seg.start))

        return place_on_timeline(clips, synth.sample_rate, out)
