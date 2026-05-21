"""Stage 6 — pipeline runner tests: all three recipes, end to end, on mocks.

Each test drives a real (tiny) video through a recipe with every swappable
backend set to its mock, then asserts on the artifact the recipe produces. The
`sample_video` fixture supplies the input and skips the suite when ffmpeg is
absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from videodub.config import (
    ASRConfig,
    SeparationConfig,
    Settings,
    TimingConfig,
    TranslationConfig,
    TTSConfig,
)
from videodub.errors import ConfigError
from videodub.media_io import probe
from videodub.pipeline import RECIPES, run_recipe
from videodub.pipeline import stages as pipeline_stages
from videodub.pipeline.context import PipelineContext
from videodub.pipeline.stages import STAGES
from videodub.schemas import Segment, Transcript
from videodub.translation.base import Translator


def _mock_settings(tmp_path: Path) -> Settings:
    """A Settings with every swappable backend set to its mock."""
    return Settings(
        work_dir=tmp_path / "work",
        asr=ASRConfig(backend="mock"),
        separation=SeparationConfig(backend="mock"),
        translation=TranslationConfig(backend="mock"),
        tts=TTSConfig(backend="mock"),
        timing=TimingConfig(backend="mock"),
    )


def test_every_recipe_stage_is_implemented():
    """Every stage name a recipe references must have a function in STAGES."""
    for stages in RECIPES.values():
        for name in stages:
            assert name in STAGES, name


def test_full_dub_produces_a_dubbed_video(sample_video: Path, tmp_path: Path):
    ctx = run_recipe(
        "full_dub", sample_video, _mock_settings(tmp_path), output=tmp_path / "out.mp4"
    )

    assert ctx.output_path.exists()
    info = probe(ctx.output_path)
    assert info.has_video and info.has_audio


def test_full_dub_writes_a_subtitle_sidecar(sample_video: Path, tmp_path: Path):
    """full_dub drops a translated .srt beside the dubbed video, sharing its stem."""
    ctx = run_recipe(
        "full_dub", sample_video, _mock_settings(tmp_path), output=tmp_path / "out.mp4"
    )

    sidecar = tmp_path / "out.srt"
    assert ctx.subtitle_path == sidecar
    assert sidecar.exists()
    text = sidecar.read_text(encoding="utf-8")
    assert "-->" in text  # has at least one cue
    assert "[en]" in text  # the mock translator tags every line with the target lang


def test_transcribe_produces_a_subtitle_file(sample_video: Path, tmp_path: Path):
    ctx = run_recipe(
        "transcribe", sample_video, _mock_settings(tmp_path), output=tmp_path / "out.srt"
    )

    assert ctx.output_path.exists()
    assert "-->" in ctx.output_path.read_text(encoding="utf-8")  # has a cue
    assert ctx.translation is None  # transcribe never translates


def test_translate_subtitles_translates_before_rendering(
    sample_video: Path, tmp_path: Path
):
    ctx = run_recipe(
        "translate_subtitles",
        sample_video,
        _mock_settings(tmp_path),
        output=tmp_path / "out.srt",
    )

    assert ctx.output_path.exists()
    # the mock translator tags every line with the target language
    assert "[en]" in ctx.output_path.read_text(encoding="utf-8")


def test_separation_self_skips_when_disabled(sample_video: Path, tmp_path: Path):
    settings = _mock_settings(tmp_path)
    settings.separation.enabled = False

    ctx = run_recipe("full_dub", sample_video, settings, output=tmp_path / "out.mp4")

    assert ctx.separated is None  # separation was skipped...
    assert ctx.output_path.exists()  # ...and the dub still completed


def test_refine_stage_overwrites_the_transcript(tmp_path: Path, monkeypatch):
    """The refine stage replaces ctx.transcript with the corrected version, so
    the subtitle and translation stages downstream see the cleaned-up text."""

    class StubTranslator(Translator):
        def translate(self, transcript, cfg):
            return transcript

        def refine(self, transcript, cfg):
            seg = transcript.segments[0]
            corrected = Segment(
                start=seg.start, end=seg.end, text="corrected", speaker=seg.speaker
            )
            return Transcript(segments=[corrected], language=transcript.language)

    monkeypatch.setattr(
        pipeline_stages, "get_translator", lambda cfg, key: StubTranslator()
    )
    ctx = PipelineContext(
        input_path=tmp_path / "in.mp4",
        output_path=tmp_path / "out.srt",
        work_dir=tmp_path,
        settings=_mock_settings(tmp_path),
    )
    ctx.transcript = Transcript(
        segments=[Segment(start=0.0, end=1.0, text="raw asr text")], language="zh"
    )

    pipeline_stages._refine(ctx)

    assert ctx.transcript.segments[0].text == "corrected"


def test_refine_self_skips_when_disabled(
    sample_video: Path, tmp_path: Path, monkeypatch
):
    """With refine_source off, the runner skips the refine stage entirely."""

    class LoudRefiner(Translator):
        def translate(self, transcript, cfg):
            return transcript

        def refine(self, transcript, cfg):
            raise AssertionError("refine ran even though refine_source is False")

    monkeypatch.setattr(
        pipeline_stages, "get_translator", lambda cfg, key: LoudRefiner()
    )
    settings = _mock_settings(tmp_path)
    settings.translation.refine_source = False

    # translate_subtitles lists `refine` before `translation`; LoudRefiner would
    # blow the run up if `refine` were not skipped.
    ctx = run_recipe(
        "translate_subtitles", sample_video, settings, output=tmp_path / "out.srt"
    )
    assert ctx.output_path.exists()


def test_refine_subtitles_recipe_rewrites_the_file(tmp_path: Path, monkeypatch):
    """`refine_subtitles` loads a subtitle file, corrects it, and overwrites it."""
    srt = tmp_path / "sub.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:02,000\n在见\n", encoding="utf-8")

    class FixingTranslator(Translator):
        def translate(self, transcript, cfg):
            return transcript

        def refine(self, transcript, cfg):
            seg = transcript.segments[0]
            fixed = Segment(
                start=seg.start, end=seg.end, text="再见", speaker=seg.speaker
            )
            return Transcript(segments=[fixed], language=transcript.language)

    monkeypatch.setattr(
        pipeline_stages, "get_translator", lambda cfg, key: FixingTranslator()
    )
    ctx = run_recipe("refine_subtitles", srt, _mock_settings(tmp_path))

    assert ctx.output_path == srt  # the input file is rewritten in place
    rewritten = srt.read_text(encoding="utf-8")
    assert "再见" in rewritten and "在见" not in rewritten


def test_refine_subtitles_ignores_the_refine_source_toggle(
    tmp_path: Path, monkeypatch
):
    """Unlike the auto-correct in other recipes, the explicit refine_subtitles
    recipe runs `refine` even when refine_source is False."""
    srt = tmp_path / "sub.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:02,000\nhello\n", encoding="utf-8")
    refined = {"called": False}

    class Recorder(Translator):
        def translate(self, transcript, cfg):
            return transcript

        def refine(self, transcript, cfg):
            refined["called"] = True
            return transcript

    monkeypatch.setattr(
        pipeline_stages, "get_translator", lambda cfg, key: Recorder()
    )
    settings = _mock_settings(tmp_path)
    settings.translation.refine_source = False
    run_recipe("refine_subtitles", srt, settings)

    assert refined["called"] is True


def test_default_output_sits_next_to_the_input(sample_video: Path, tmp_path: Path):
    ctx = run_recipe("transcribe", sample_video, _mock_settings(tmp_path))

    assert ctx.output_path == sample_video.with_name(f"{sample_video.stem}.srt")
    assert ctx.output_path.exists()


def test_keep_intermediates_false_removes_work_dir(
    sample_video: Path, tmp_path: Path
):
    settings = _mock_settings(tmp_path)
    settings.keep_intermediates = False

    ctx = run_recipe(
        "transcribe", sample_video, settings, output=tmp_path / "out.srt"
    )

    assert ctx.output_path.exists()  # the output survives the cleanup...
    assert not ctx.work_dir.exists()  # ...but the intermediates are gone


def test_unknown_recipe_raises(sample_video: Path, tmp_path: Path):
    with pytest.raises(ConfigError, match="unknown recipe"):
        run_recipe("nope", sample_video, _mock_settings(tmp_path))


def test_missing_input_raises(tmp_path: Path):
    with pytest.raises(ConfigError, match="input file not found"):
        run_recipe("transcribe", tmp_path / "ghost.mp4", _mock_settings(tmp_path))
