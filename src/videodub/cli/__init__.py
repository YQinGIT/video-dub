"""videodub — command-line interface.

Commands:
    videodub run <recipe> <input>   run any recipe
    videodub recipes                list the available recipes
    videodub transcribe <input>     shortcut for `run transcribe`
    videodub subtitle <input>       shortcut for `run translate_subtitles`
    videodub dub <input>            shortcut for `run full_dub`
    videodub refine <subtitle>      shortcut for `run refine_subtitles`

Backends are chosen by config: pass `--config FILE` (TOML or JSON) or set
`VIDEODUB_*` environment variables. With no config the defaults select the real
GPU / API backends; `recipes/mock.toml` selects the all-mock path.

This module deliberately does not use `from __future__ import annotations`:
Typer inspects the real annotation objects to build the CLI.
"""

import json
import logging
import tomllib
from pathlib import Path

import typer

from videodub.config import Settings
from videodub.errors import ConfigError, VideodubError
from videodub.pipeline import RECIPES, run_recipe

app = typer.Typer(
    help="Local-first, modular video localization toolkit.",
    no_args_is_help=True,
    add_completion=False,
)


def _load_settings(config: Path | None) -> Settings:
    """Build `Settings`, optionally overlaying a TOML or JSON config file.

    Values in the file win over environment variables; any field the file omits
    still falls back to the environment, `.env`, then the defaults — so secrets
    keep coming from the environment even when `--config` is passed.
    """
    if config is None:
        return Settings()

    text = config.read_text(encoding="utf-8")
    if config.suffix == ".toml":
        data = tomllib.loads(text)
    elif config.suffix == ".json":
        data = json.loads(text)
    else:
        raise ConfigError(
            f"unsupported config format {config.suffix!r}; use .toml or .json"
        )
    return Settings(**data)


def _run(
    recipe: str, input_path: Path, config: Path | None, output: Path | None
) -> None:
    """Shared body of `run` and the three convenience commands."""
    try:
        settings = _load_settings(config)
        ctx = run_recipe(recipe, input_path, settings, output=output)
    except VideodubError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.secho(f"done: {ctx.output_path}", fg=typer.colors.GREEN)
    if ctx.subtitle_path is not None:
        typer.secho(
            f"      subtitles: {ctx.subtitle_path}", fg=typer.colors.GREEN
        )


@app.callback()
def _configure() -> None:
    """Configure logging before any command runs."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@app.command()
def recipes() -> None:
    """List the available pipeline recipes."""
    for name, stages in RECIPES.items():
        typer.echo(f"{name:<20} {' -> '.join(stages)}")


@app.command()
def run(
    recipe: str = typer.Argument(..., help="Recipe name; see `videodub recipes`."),
    input_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Input video file."
    ),
    config: Path | None = typer.Option(
        None, "--config", "-c", exists=True, dir_okay=False,
        help="TOML or JSON config file selecting backends.",
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output path (defaults next to the input)."
    ),
) -> None:
    """Run any recipe on an input file."""
    _run(recipe, input_path, config, output)


@app.command()
def transcribe(
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    config: Path | None = typer.Option(
        None, "--config", "-c", exists=True, dir_okay=False
    ),
    output: Path | None = typer.Option(None, "--output", "-o"),
) -> None:
    """Transcribe a video to a subtitle file (recipe: transcribe)."""
    _run("transcribe", input_path, config, output)


@app.command()
def subtitle(
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    config: Path | None = typer.Option(
        None, "--config", "-c", exists=True, dir_okay=False
    ),
    output: Path | None = typer.Option(None, "--output", "-o"),
) -> None:
    """Produce translated subtitles for a video (recipe: translate_subtitles)."""
    _run("translate_subtitles", input_path, config, output)


@app.command()
def dub(
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    config: Path | None = typer.Option(
        None, "--config", "-c", exists=True, dir_okay=False
    ),
    output: Path | None = typer.Option(None, "--output", "-o"),
) -> None:
    """Produce a voice-preserving dub of a video (recipe: full_dub)."""
    _run("full_dub", input_path, config, output)


@app.command()
def refine(
    input_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Subtitle file (.srt or .vtt)."
    ),
    config: Path | None = typer.Option(
        None, "--config", "-c", exists=True, dir_okay=False
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o",
        help="Where to write the corrected file (default: overwrite the input).",
    ),
) -> None:
    """Proofread a subtitle file's ASR text (recipe: refine_subtitles).

    The subtitle text is sent to DeepSeek, which fixes speech-recognition
    errors. With no --output the corrected file replaces the input.
    """
    _run("refine_subtitles", input_path, config, output)


def main() -> None:
    """Console-script entry point (`videodub` on the command line)."""
    app()
