"""Stage 6 — CLI tests, driven through Typer's CliRunner.

The CLI is exercised in-process; no `videodub` console script needs to be
installed. `VIDEODUB_WORK_DIR` is set per invocation so intermediates land in
the test's tmp directory rather than `./workdir`.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from videodub.cli import app

runner = CliRunner()

# The all-mock config shipped in the repo, so the CLI runs with no GPU.
_MOCK_CONFIG = Path(__file__).resolve().parents[1] / "recipes" / "mock.toml"


def test_recipes_command_lists_every_recipe():
    result = runner.invoke(app, ["recipes"])

    assert result.exit_code == 0
    for name in ("full_dub", "translate_subtitles", "transcribe", "refine_subtitles"):
        assert name in result.stdout


def test_run_transcribe_with_mock_config(sample_video: Path, tmp_path: Path):
    output = tmp_path / "out.srt"
    result = runner.invoke(
        app,
        ["run", "transcribe", str(sample_video),
         "--config", str(_MOCK_CONFIG), "--output", str(output)],
        env={"VIDEODUB_WORK_DIR": str(tmp_path / "work")},
    )

    assert result.exit_code == 0, result.stdout
    assert output.exists()


def test_dub_convenience_command(sample_video: Path, tmp_path: Path):
    output = tmp_path / "out.mp4"
    result = runner.invoke(
        app,
        ["dub", str(sample_video),
         "--config", str(_MOCK_CONFIG), "--output", str(output)],
        env={"VIDEODUB_WORK_DIR": str(tmp_path / "work")},
    )

    assert result.exit_code == 0, result.stdout
    assert output.exists()


def test_refine_convenience_command(tmp_path: Path):
    # `refine` takes a subtitle file, not a video; the mock backend leaves the
    # text untouched, so the run just needs to succeed and keep the file.
    srt = tmp_path / "sub.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:02,000\n你好\n", encoding="utf-8")
    result = runner.invoke(
        app,
        ["refine", str(srt), "--config", str(_MOCK_CONFIG)],
        env={"VIDEODUB_WORK_DIR": str(tmp_path / "work")},
    )

    assert result.exit_code == 0, result.stdout
    assert "你好" in srt.read_text(encoding="utf-8")


def test_unknown_recipe_exits_nonzero(sample_video: Path, tmp_path: Path):
    result = runner.invoke(
        app,
        ["run", "bogus", str(sample_video), "--config", str(_MOCK_CONFIG)],
        env={"VIDEODUB_WORK_DIR": str(tmp_path / "work")},
    )

    assert result.exit_code == 1


def test_missing_input_file_is_rejected(tmp_path: Path):
    # Typer's `exists=True` on the argument rejects the path before the run.
    result = runner.invoke(
        app, ["transcribe", str(tmp_path / "ghost.mp4"), "--config", str(_MOCK_CONFIG)]
    )

    assert result.exit_code != 0
