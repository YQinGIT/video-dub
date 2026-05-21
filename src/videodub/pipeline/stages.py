"""Thin wrappers that bind each stage module to the pipeline context.

Every function here takes the shared `PipelineContext`, does one stage's work
by calling that stage's module, and writes its result back onto the context.
`STAGES` maps the stage names used in `recipes.py` to these functions.

This is the one module allowed to import every stage — it is the orchestrator's
glue, not a peer stage. Each stage module's package imports cheaply (the heavy
backends load lazily inside the factories), so importing this module pulls in
no GPU library.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from videodub import media_io
from videodub.asr import get_asr_backend
from videodub.errors import StageError
from videodub.mixing import mix
from videodub.pipeline.context import PipelineContext
from videodub.schemas import Transcript
from videodub.separation import get_separator
from videodub.subtitle import load as load_subtitle
from videodub.subtitle import write as write_subtitle
from videodub.timing import get_timing_fitter
from videodub.translation import get_translator
from videodub.tts import get_tts_backend, trim_silence


def _require_audio(ctx: PipelineContext) -> Path:
    """The extracted audio, or a StageError if `extract_audio` has not run."""
    if ctx.audio is None:
        raise StageError("a stage needs extracted audio, but extract_audio has not run")
    return ctx.audio


def _require_transcript(ctx: PipelineContext) -> Transcript:
    """The ASR transcript, or a StageError if `asr` has not run."""
    if ctx.transcript is None:
        raise StageError("a stage needs a transcript, but asr has not run")
    return ctx.transcript


def _extract_audio(ctx: PipelineContext) -> None:
    ctx.audio = media_io.extract_audio(ctx.input_path, ctx.path("audio.wav"))


def _separation(ctx: PipelineContext) -> None:
    backend = get_separator(ctx.settings.separation)
    ctx.separated = backend.separate(
        _require_audio(ctx), ctx.settings.separation, ctx.work_dir / "separation"
    )


def _asr(ctx: PipelineContext) -> None:
    backend = get_asr_backend(ctx.settings.asr)
    # Transcribe the isolated vocals when separation ran, the raw audio if not.
    audio = ctx.separated.vocals if ctx.separated else _require_audio(ctx)
    ctx.transcript = backend.transcribe(audio, ctx.settings.asr)


def _load_subtitle(ctx: PipelineContext) -> None:
    # The refine_subtitles recipe starts from an existing subtitle file, not a
    # video — parse it into a transcript for the refine stage to proofread.
    ctx.transcript = load_subtitle(
        ctx.input_path, language=ctx.settings.translation.source_language
    )


def _refine(ctx: PipelineContext) -> None:
    backend = get_translator(ctx.settings.translation, ctx.settings.deepseek_api_key)
    # Overwrite the transcript in place: the ASR output is replaced by the
    # proofread version, so every later stage — subtitle rendering and
    # translation alike — works from the corrected source text.
    ctx.transcript = backend.refine(
        _require_transcript(ctx), ctx.settings.translation
    )


def _translation(ctx: PipelineContext) -> None:
    backend = get_translator(ctx.settings.translation, ctx.settings.deepseek_api_key)
    ctx.translation = backend.translate(
        _require_transcript(ctx), ctx.settings.translation
    )


def _tts(ctx: PipelineContext) -> None:
    backend = get_tts_backend(ctx.settings.tts)
    # The dub speaks the translation; fall back to the transcript if a recipe
    # ever runs tts without a translation step.
    transcript = ctx.translation or _require_transcript(ctx)
    # Voice-cloning reference: an explicit clip from config when set, otherwise
    # the isolated vocals from the separation stage. Real backends require one;
    # the mock ignores it.
    reference = ctx.settings.tts.reference_audio
    if reference is None and ctx.separated is not None:
        reference = ctx.separated.vocals
    synthesized = backend.synthesize(
        transcript, ctx.settings.tts, ctx.work_dir / "tts", reference_audio=reference
    )
    # A neural TTS clip can carry dead air the model invented — long internal
    # pauses, leading padding — which would force the timing stage to cut speech
    # to make the clip fit its slot. Strip it now so timing only ever stretches.
    if ctx.settings.tts.trim_silence:
        synthesized = trim_silence(synthesized, ctx.work_dir / "tts" / "_trimmed")
    ctx.synthesized = synthesized


def _timing(ctx: PipelineContext) -> None:
    if ctx.synthesized is None:
        raise StageError("timing stage ran before tts produced any audio")
    backend = get_timing_fitter(ctx.settings.timing)
    ctx.fitted_vocals = backend.fit(
        ctx.synthesized, ctx.settings.timing, ctx.path("dubbed_vocals.wav")
    )


def _mixing(ctx: PipelineContext) -> None:
    if ctx.fitted_vocals is None:
        raise StageError("mixing stage ran before timing produced a vocal track")
    if ctx.separated is None:
        # Separation was skipped -> there is no preserved background, so the
        # dub is the vocal track alone. Nothing to mix.
        ctx.mixed_audio = ctx.fitted_vocals
        return
    ctx.mixed_audio = mix(
        ctx.fitted_vocals,
        ctx.separated.background,
        ctx.path("mixed.wav"),
        ctx.settings.mixing,
    )


def _remux(ctx: PipelineContext) -> None:
    if ctx.mixed_audio is None:
        raise StageError("remux stage ran before mixing produced any audio")
    media_io.remux(ctx.input_path, ctx.mixed_audio, ctx.output_path)


def _subtitle(ctx: PipelineContext) -> None:
    # Render the translation when there is one, otherwise the raw transcript.
    transcript = ctx.translation or _require_transcript(ctx)
    write_subtitle(transcript, ctx.output_path)


def _subtitle_sidecar(ctx: PipelineContext) -> None:
    # full_dub writes the dubbed video to ctx.output_path; this drops a matching
    # subtitle file beside it — same stem, `.srt` extension — so a media player
    # auto-loads it (luoxiang.dubbed.mp4 -> luoxiang.dubbed.srt). The cues carry
    # the translated lines on the source segment timings, the same slots the
    # timing stage fits the dubbed audio to, so the text tracks the speech.
    transcript = ctx.translation or _require_transcript(ctx)
    ctx.subtitle_path = write_subtitle(
        transcript, ctx.output_path.with_suffix(".srt")
    )


STAGES: dict[str, Callable[[PipelineContext], None]] = {
    "extract_audio": _extract_audio,
    "separation": _separation,
    "asr": _asr,
    "load_subtitle": _load_subtitle,
    "refine": _refine,
    "translation": _translation,
    "tts": _tts,
    "timing": _timing,
    "mixing": _mixing,
    "remux": _remux,
    "subtitle": _subtitle,
    "subtitle_sidecar": _subtitle_sidecar,
}
