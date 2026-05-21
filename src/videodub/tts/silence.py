"""Trim dead air out of synthesized speech clips — a TTS post-process. PORTABLE.

A neural TTS model does not always render a line as tightly as a person would
speak it. IndexTTS-2 in particular can pad a short utterance with a long pause
in the middle, or chop it into bursts with gaps between them. The clip then runs
far longer than the timeline slot it must fill, and the `timing` stage — which
can only compress a clip so far before speech sounds unnatural — is forced to
cut speech off the end to make it fit.

This module removes that dead air *before* the timing stage runs. It shells out
to ffmpeg's `silenceremove` filter to drop leading silence outright and cap
every internal or trailing pause at `_MAX_GAP_S` seconds. What comes out is a
clip whose length is essentially its speech content, so the timing stage only
has to stretch it gently and never has to trim a word.

Capping each pause — rather than removing silence wholesale — is deliberate: a
real pause between clauses survives as a short, natural gap; only the dead air
*beyond* the cap, which is the model's invented padding, is cut.

Like the other audio helpers this is portable: it needs the ffmpeg binary, no
GPU and no network.
"""

from __future__ import annotations

from pathlib import Path

from videodub._ffmpeg import audio_duration, run_ffmpeg
from videodub.errors import BackendError
from videodub.schemas import SynthesizedAudio, SynthSegment

# A sample quieter than this many dB below full scale counts as silence.
_THRESHOLD_DB = 40.0
# Every detected pause is capped to this many seconds; dead air past it is cut.
# 0.15s reads as a natural short pause — long enough that clauses do not slur
# together, short enough to reclaim almost all of a model's invented padding.
_MAX_GAP_S = 0.15


def _silenceremove_filter() -> str:
    """The ffmpeg `silenceremove` filtergraph: strip the lead-in, cap every gap.

    `start_periods=1` removes the silence before speech begins; `stop_periods=-1`
    then acts on every later silence — internal *and* trailing — keeping only
    `stop_duration` seconds of each and cutting the rest. `detection=peak` judges
    silence by peak level, which is conservative: a faint sound spares a gap.
    """
    threshold = f"-{_THRESHOLD_DB:g}dB"
    return (
        "silenceremove="
        f"start_periods=1:start_threshold={threshold}:"
        f"stop_periods=-1:stop_threshold={threshold}:"
        f"stop_duration={_MAX_GAP_S}:detection=peak"
    )


def strip_silence(src: Path, dst: Path) -> Path:
    """Write `src` to `dst` with leading silence dropped and every pause capped.

    Returns `dst`. Raises `BackendError` if ffmpeg is missing or exits non-zero.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(["-i", str(src), "-af", _silenceremove_filter(), str(dst)])
    return dst


def trim_silence(synth: SynthesizedAudio, out_dir: Path) -> SynthesizedAudio:
    """Return a copy of `synth` whose every clip has had its dead air trimmed.

    Each clip is rewritten under `out_dir`; the `SynthSegment` metadata — slot
    timestamps, text, speaker — carries over unchanged, only `audio_path` moves
    to the trimmed file. A clip that trims away to nothing (which a real speech
    clip never should) keeps its original file, so the contract that every
    segment points at a usable clip always holds.
    """
    out_dir = Path(out_dir)
    if not synth.segments:
        return synth
    out_dir.mkdir(parents=True, exist_ok=True)

    trimmed: list[SynthSegment] = []
    for index, seg in enumerate(synth.segments):
        src = Path(seg.audio_path)
        dst = strip_silence(src, out_dir / f"segment_{index:04d}.wav")
        # Guard the degenerate case: if trimming emptied the clip, fall back to
        # the original so the segment still points at audio the timing stage
        # can use. `audio_duration` raises on a zero-length file, hence the try.
        try:
            emptied = audio_duration(dst) <= 0
        except BackendError:
            emptied = True
        path = src if emptied else dst
        trimmed.append(seg.model_copy(update={"audio_path": path}))

    return SynthesizedAudio(segments=trimmed, sample_rate=synth.sample_rate)
