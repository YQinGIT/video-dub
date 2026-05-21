# video-dub

Local-first, modular video localization toolkit. Takes a video in one language and
produces a dubbed video in another while **preserving the original speaker's voice**.
Primary use case: Chinese video → English dub.

Each stage (media I/O, separation, ASR, translation, subtitles, TTS, timing, mixing) is
an independent module behind a typed Pydantic contract, composable into named
**recipes** — full dub, translated subtitles, or transcribe-only.

## Status

Under active construction, stage by stage. See **[PLAN.md](PLAN.md)** for the full build
plan and the progress tracker. A fresh session should read `PLAN.md` first and resume at
the last unchecked stage.

## Requirements

- Python 3.12 (managed by [`uv`](https://docs.astral.sh/uv/))
- `ffmpeg` on `PATH`
- An NVIDIA GPU with CUDA for the local ASR / separation / TTS stages. The portable path
  runs anywhere on mock backends.

## Setup

```bash
uv sync           # core (portable) dependencies
uv run pytest     # run the test suite (CUDA-free)
```

The GPU stack (`uv sync --extra gpu`) is added in Stage 7.
