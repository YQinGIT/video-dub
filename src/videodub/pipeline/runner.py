"""Walk a recipe's stage list, running each stage against a shared context.

`run_recipe` is the pipeline's single entry point. It builds a `PipelineContext`,
runs the recipe's stages in order, skips the conditional ones, and returns the
completed context. The CLI and (later) the service both call straight into it.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from videodub.config import Settings
from videodub.errors import ConfigError, StageError, VideodubError
from videodub.pipeline.context import PipelineContext
from videodub.pipeline.recipes import RECIPES
from videodub.pipeline.stages import STAGES

logger = logging.getLogger("videodub.pipeline")

# Conditional stages — toggles, not recipes of their own: wherever a recipe
# lists one, the runner skips it when its config switch is off. `separation`
# follows `SeparationConfig.enabled`. `refine` (the ASR-correction pass) follows
# `TranslationConfig.refine_source` — except in `refine_subtitles`, the recipe
# whose entire job is to refine, where it always runs.
_SKIP_WHEN = {
    "separation": lambda s, recipe: not s.separation.enabled,
    "refine": lambda s, recipe: (
        recipe != "refine_subtitles" and not s.translation.refine_source
    ),
}

# The output file extension each recipe uses when the caller passes no `output`.
# `refine_subtitles` is absent: it rewrites the file it was given (see below).
_DEFAULT_OUTPUT_SUFFIX = {
    "full_dub": ".dubbed.mp4",
    "transcribe": ".srt",
    "translate_subtitles": ".translated.srt",
}


def default_output(recipe: str, input_path: Path) -> Path:
    """The path a recipe writes to when the caller passes no `output`.

    It sits next to the input file — and so outside the work directory, which
    means it survives the `keep_intermediates=False` cleanup. `refine_subtitles`
    is the exception: it overwrites the subtitle file it was handed.
    """
    if recipe == "refine_subtitles":
        return input_path
    return input_path.with_name(input_path.stem + _DEFAULT_OUTPUT_SUFFIX[recipe])


def run_recipe(
    recipe: str,
    input_path: Path | str,
    settings: Settings,
    output: Path | str | None = None,
) -> PipelineContext:
    """Run `recipe` on `input_path` and return the completed `PipelineContext`.

    `output` defaults to a path next to the input (see `default_output`).
    Intermediate files go under `settings.work_dir`; they are deleted afterwards
    unless `settings.keep_intermediates` is True.

    Raises `ConfigError` for an unknown recipe or a missing input, and
    `StageError` if a stage fails.
    """
    if recipe not in RECIPES:
        raise ConfigError(
            f"unknown recipe {recipe!r}; choose from {', '.join(sorted(RECIPES))}"
        )

    input_path = Path(input_path)
    if not input_path.exists():
        raise ConfigError(f"input file not found: {input_path}")

    output = (
        Path(output) if output is not None else default_output(recipe, input_path)
    )
    work_dir = settings.work_dir / input_path.stem
    work_dir.mkdir(parents=True, exist_ok=True)

    ctx = PipelineContext(
        input_path=input_path,
        output_path=output,
        work_dir=work_dir,
        settings=settings,
    )

    stage_names = RECIPES[recipe]
    logger.info("recipe %r: %s", recipe, " -> ".join(stage_names))
    for name in stage_names:
        skip = _SKIP_WHEN.get(name)
        if skip is not None and skip(settings, recipe):
            logger.info("  skip   %s", name)
            continue
        logger.info("  run    %s", name)
        _run_stage(name, ctx)

    if not settings.keep_intermediates:
        shutil.rmtree(work_dir, ignore_errors=True)
        logger.info("removed intermediates: %s", work_dir)

    return ctx


def _run_stage(name: str, ctx: PipelineContext) -> None:
    """Run one stage, turning any non-videodub error into a `StageError`."""
    try:
        STAGES[name](ctx)
    except VideodubError:
        raise  # already a clear, typed error — let it through unchanged
    except Exception as exc:
        raise StageError(f"stage {name!r} failed: {exc}") from exc
