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
from videodub.pipeline.stages import STAGES


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
