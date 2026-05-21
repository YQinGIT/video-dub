# Video-Dub — Build Plan

> **Persistence note:** The very first action on approval (Stage 0) is to copy this
> document to `PLAN.md` in the repo root and keep its **Progress Tracker** updated as
> stages complete. A fresh Claude Code session should read `PLAN.md` first, find the
> last unchecked stage, and resume there.

---

## Context

We are building a **local-first, modular video localization toolkit**. Primary use
case: Chinese video → English dub that **preserves the original speaker's voice**. The
deeper goal is a **modular toolkit** where each stage is reusable and composable into
named "recipes" (e.g. transcribe-only, or translated subtitles for a foreign film with
no voice synthesis at all). Modularity is a first-class requirement.

Cost is driven near zero by running everything **locally on an RTX 5080 (16 GB,
Blackwell, CUDA)**, except translation which uses the **DeepSeek API** (near-free,
better context handling than a small local model; backend kept swappable). Stages run
**sequentially**, so all models never need to be in VRAM at once. **Lip-sync is out of
scope for v1** — only a clean interface stub is left for it.

The work is built **stage by stage** (not all at once) and **portable-first**: every
GPU stage has a mock backend so the full pipeline and its tests run end-to-end on any
machine with no CUDA.

---

## Environment status — DONE

- WSL2 (Ubuntu) on the Windows GPU box. We develop and run **inside WSL2**; the Mac
  SSHes in. No code runs on the Mac anymore.
- Verified on PATH: `nvidia-smi` (RTX 5080, 16 GB, driver 596.36), `uv` 0.11.15,
  `ffmpeg` 8.0.1, `python3`.
- **Decision:** project pins **Python 3.12** via `uv` (system Python is 3.14, too new
  for the CUDA ML stack — torch cu128 / ctranslate2 / demucs wheels lag a release).

---

## Architecture principles (non-negotiable)

1. **Pydantic** for every data contract and all config.
2. Every swappable backend (ASR engine, translator, TTS model, separator, timing
   fitter) sits behind an **ABC / Protocol**. Implementation chosen by a **config
   string**.
3. **Lazy imports**: each backend module's `__init__.py` exposes a factory that imports
   the heavy implementation *inside the function*, only for the selected backend. So
   `import videodub.asr` never pulls `torch`. **No central registry** (it would
   transitively import every GPU library).
4. **No cross-stage imports** — modules communicate only through `schemas.py`.
5. **A mock backend for every GPU stage** — the whole pipeline is testable with no CUDA.
6. Secrets (API keys) only via env / `.env` as `SecretStr`. Never hard-coded.
7. `pytest` stays **green and CUDA-free at every stage**. Real GPU backends get
   separate `@pytest.mark.gpu` tests, skipped when CUDA is absent.

---

## Approved data contract — `src/videodub/schemas.py`

```python
from __future__ import annotations
from pydantic import BaseModel, Field, model_validator


class Segment(BaseModel):
    """One timed unit of speech — the atom of a Transcript."""
    start: float = Field(ge=0, description="Start time, seconds")
    end:   float = Field(ge=0, description="End time, seconds")
    text:  str   = Field(description="Text for this segment")
    speaker: str | None = Field(default=None, description="Diarization label, if any")

    @property
    def duration(self) -> float:
        return self.end - self.start

    @model_validator(mode="after")
    def _ordered(self) -> "Segment":
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) precedes start ({self.start})")
        return self


class Transcript(BaseModel):
    """Ordered timed segments. The single object that flows
    ASR -> translation -> (subtitle | TTS)."""
    segments: list[Segment] = Field(default_factory=list)
    language: str | None = Field(default=None,
        description="ISO code of the text currently in `segments`")
    source_language: str | None = Field(default=None,
        description="Original language, set when this is a translation")

    @property
    def duration(self) -> float:
        return self.segments[-1].end if self.segments else 0.0
```

**Contract decisions:**
- `Segment` stays minimal. Translation returns a **new** `Transcript`, same timestamps,
  `text` replaced; translation is contractually **1:1 and timestamp-preserving**, so
  list index aligns source ↔ translated segments (no `id` field needed).
- **Bilingual subtitles (approved):** the `subtitle` renderer accepts an optional
  **secondary `Transcript`** and stacks both languages per cue (original + translation).
  Single-language is the default; bilingual is opt-in. No schema pollution.
- TTS output (per-segment synthesized audio) is **not** a `Transcript` — a separate
  artifact type (`SynthSegment` / `SynthesizedAudio`) defined when the `tts` stage is
  built.

---

## Approved config — `src/videodub/config.py`

```python
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ASRConfig(BaseModel):
    backend: Literal["faster_whisper", "mock"] = "faster_whisper"
    model_size: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str | None = None        # None -> autodetect
    diarize: bool = False

class TranslationConfig(BaseModel):
    backend: Literal["deepseek", "ollama", "mock"] = "deepseek"
    model: str = "deepseek-chat"
    source_language: str = "zh"
    target_language: str = "en"
    timing_aware: bool = True          # prompt model to hit segment durations

class SeparationConfig(BaseModel):
    backend: Literal["demucs", "mock"] = "demucs"
    enabled: bool = True               # toggle the whole stage
    device: str = "cuda"

class TTSConfig(BaseModel):
    backend: Literal["cosyvoice2", "gpt_sovits", "elevenlabs", "mock"] = "cosyvoice2"
    device: str = "cuda"
    reference_audio: Path | None = None  # None -> clone from source vocals

class TimingConfig(BaseModel):
    backend: Literal["rubberband", "mock"] = "rubberband"
    max_stretch: float = 1.3           # cap so the dub stays natural
    min_stretch: float = 0.7

class MixingConfig(BaseModel):
    background_gain_db: float = 0.0
    vocal_gain_db: float = 0.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VIDEODUB_", env_file=".env",
        env_nested_delimiter="__", extra="ignore",
    )
    work_dir: Path = Path("./workdir")
    keep_intermediates: bool = True

    deepseek_api_key:   SecretStr | None = None
    elevenlabs_api_key: SecretStr | None = None

    asr:         ASRConfig         = ASRConfig()
    translation: TranslationConfig = TranslationConfig()
    separation:  SeparationConfig  = SeparationConfig()
    tts:         TTSConfig         = TTSConfig()
    timing:      TimingConfig      = TimingConfig()
    mixing:      MixingConfig      = MixingConfig()
```

Nested overrides work via env (`VIDEODUB_ASR__BACKEND=mock`) or a config file passed to
the CLI — so swapping mock ↔ real backends is **config only**, no code change.

---

## Repo structure

```
video-dub/
├── PLAN.md                   # this document, copied in at Stage 0; progress tracked
├── pyproject.toml            # uv-managed; core deps + optional [gpu] extra
├── .python-version           # 3.12
├── .env.example
├── .gitignore
├── README.md
├── src/videodub/
│   ├── __init__.py
│   ├── schemas.py            # Transcript / Segment — shared contract
│   ├── config.py             # Pydantic Settings, per-stage config
│   ├── errors.py             # shared exception types
│   ├── media_io/             # ffmpeg wrappers (subprocess)        — PORTABLE
│   ├── separation/           # Demucs vocal/bg split               — CUDA-BOUND
│   ├── asr/                  # audio -> Transcript                 — CUDA-BOUND
│   ├── translation/          # Transcript -> translated Transcript — PORTABLE
│   ├── subtitle/             # Transcript -> SRT/VTT/ASS           — PORTABLE
│   ├── tts/                  # translated Transcript -> speech     — CUDA-BOUND
│   ├── timing/               # fit segments to durations           — PORTABLE*
│   ├── mixing/               # vocals + background -> final audio  — PORTABLE
│   ├── lipsync/              # v1 STUB / interface only
│   ├── pipeline/             # orchestration + recipe definitions
│   ├── service/              # FastAPI app over the LAN
│   └── cli/                  # Typer CLI
├── tests/                    # mirrors src/; runs fully on mocks, no CUDA
│   ├── conftest.py           # fixtures: sample Transcript, tiny generated media
│   └── data/
└── recipes/                  # example config files per recipe
```

`* timing` is CPU code but depends on the `rubberband` system binary.

**Per-module layout** (any stage with swappable backends — `asr`, `translation`,
`separation`, `tts`, `timing`):

```
asr/
├── __init__.py        # get_asr_backend(cfg) -> ASRBackend   (lazy-import factory)
├── base.py            # ASRBackend  (ABC / Protocol)
├── faster_whisper.py  # CUDA backend — heavy imports happen HERE
└── mock.py            # mock backend — deterministic, no GPU
```

**Recipes** are pure data in `pipeline/recipes.py` — an ordered list of stage names, no
recipe-specific logic:

```python
RECIPES = {
    "full_dub":            ["extract_audio", "separation", "asr", "translation",
                            "tts", "timing", "mixing", "remux"],
    "translate_subtitles": ["extract_audio", "asr", "translation", "subtitle"],
    "transcribe":          ["extract_audio", "asr", "subtitle"],
}
```
`separation` self-skips when `enabled=False` (a toggle, not a separate recipe).

---

## Dependencies (`uv`)

- **Core (portable, installs anywhere):** `pydantic`, `pydantic-settings`, `httpx`
  (DeepSeek), `typer` (CLI), `fastapi` + `uvicorn` (service), `pytest`, `pytest-cov`,
  `ruff`. `ffmpeg` and `rubberband` are **system binaries** called via subprocess — no
  Python bindings.
- **`[gpu]` extra (CUDA box only):** `torch` (cu128 / Blackwell `sm_120`),
  `faster-whisper`, `demucs`; TTS deps added at Stage 7c. `uv sync` installs core;
  `uv sync --extra gpu` adds the GPU stack.
- The SRT/VTT/ASS renderer is hand-written — trivial, avoids a dependency.
- **Rule:** ask before adding any dependency not listed here.

---

## Staged build plan

Each stage is a self-contained increment. `uv run pytest` and `uv run ruff check` must
pass at the end of every stage. Pause for review at each stage boundary.

### Stage 0 — Scaffold & tooling  *(PORTABLE)*
- `uv python pin 3.12`; create `pyproject.toml` (project metadata, core deps, `[gpu]`
  extra, ruff + pytest config, `videodub` console script entry point).
- Create `.python-version`, `.gitignore`, `.env.example`, `README.md`, and **copy this
  plan to `PLAN.md`**.
- Create `src/videodub/` package tree — every module dir with `__init__.py`.
- `uv sync`; verify the environment.
- **DoD:** `uv run python -c "import videodub"` works; `uv run pytest` collects cleanly;
  `uv run ruff check` passes.

### Stage 1 — Core contracts  *(PORTABLE)*
- `schemas.py` (Segment, Transcript as above; plus small helpers if needed).
- `config.py` (Settings + per-stage configs, env loading).
- `errors.py` (base exceptions: `ConfigError`, `StageError`, `BackendError`).
- `tests/`: schema validation (ordering validator, `duration` props), config env
  override (`VIDEODUB_ASR__BACKEND`), `SecretStr` handling.
- **DoD:** tests pass.

### Stage 2 — `media_io`  *(PORTABLE — ffmpeg)*
- ffmpeg/ffprobe subprocess wrappers: `extract_audio(video, out, sr=16000, mono=True)`,
  `remux(video, audio, out)`, `probe(path) -> MediaInfo`.
- `MediaInfo` Pydantic model (duration, streams, codecs).
- `tests/`: generate a tiny clip in `conftest.py` via ffmpeg (`testsrc` + `sine`),
  then extract / probe / remux and assert.
- **DoD:** tests pass on this box.

### Stage 3 — `subtitle`  *(PORTABLE)*
- Renderer for SRT / VTT / ASS: `render(transcript, fmt, secondary=None) -> str` plus a
  file writer. `secondary` enables **bilingual stacked** output.
- Per-format timestamp formatting.
- `tests/`: golden-string render of a known Transcript to each format; bilingual case.
- **DoD:** tests pass.

### Stage 4 — `translation`  *(PORTABLE — DeepSeek API)*
- `base.py`: `Translator` ABC — `translate(transcript, cfg) -> Transcript`.
- `deepseek.py`: `httpx` client, OpenAI-compatible endpoint; **timing-aware prompt**
  (passes per-segment durations / char budgets so the dub does not drift); segment
  batching; structured JSON output; retry/backoff. Key from `Settings.deepseek_api_key`.
- `mock.py`: deterministic offline translator.
- `__init__.py`: `get_translator(cfg)` factory.
- `tests/`: mock translator; DeepSeek client with **mocked httpx** (no real network);
  prompt-construction unit test. Optional manual smoke script gated by an env key.
- **DoD:** tests pass.

### Stage 5 — Mock GPU backends  *(PORTABLE)*
- `asr/base.py` `ASRBackend` ABC + `asr/mock.py` (deterministic Transcript).
- `separation/base.py` `Separator` ABC + `separation/mock.py` (vocals = input copy,
  background = silence).
- `tts/base.py` `TTSBackend` ABC + output artifact schema + `tts/mock.py` (per-segment
  silence/sine at target duration via ffmpeg).
- `timing/base.py` `TimingFitter` ABC + `timing/mock.py` (concat/pad, no real stretch).
- Each module's `__init__.py` lazy-import factory.
- `tests/`: each mock backend; factory selection by config string.
- **DoD:** tests pass; the whole portable path imports with no CUDA.

### Stage 6 — `pipeline` + `cli` + `mixing`  *(PORTABLE)*
- `mixing/`: `mix(vocals, background, cfg) -> audio` (ffmpeg `amix`).
- `pipeline/recipes.py` (RECIPES data); `pipeline/context.py` (`PipelineContext` — typed
  Pydantic artifact bag); `pipeline/stages.py` (thin stage wrappers binding modules to
  context); `pipeline/runner.py` (walks the recipe, handles skip/intermediates/logging).
- `cli/`: Typer app — `videodub run <recipe> <input> [--config ...]`,
  `videodub recipes`, plus `transcribe` / `subtitle` / `dub` convenience commands.
- `tests/`: all three recipes end-to-end with an all-mock config.
- **DoD:** `uv run videodub run full_dub <sample>` produces a dubbed file on mocks; all
  three recipes green.

### Stage 7 — Real CUDA backends  *(CUDA-BOUND — one at a time, verified on GPU)*
Install `[gpu]` extra (`uv sync --extra gpu`: torch cu128, faster-whisper, demucs).
- **7a `asr/faster_whisper.py`** — faster-whisper large-v3, CUDA. Verify on real audio.
  Diarization deferred (WhisperX / pyannote noted as a later option).
- **7b `separation/demucs.py`** — Demucs (htdemucs). Verify vocal/background split.
- **7c `tts/cosyvoice2.py`** — CosyVoice 2 (default; strong zh + cross-lingual cloning).
  Heaviest install — may need vendoring; **ask before adding its deps**. Verify zh→en
  voice cloning. GPT-SoVITS noted as alternate backend.
- **7d `timing/rubberband.py`** — `pyrubberband` + `rubberband` CLI; time-stretch within
  `min/max_stretch`, pad/trim silence, assemble a continuous vocal track.
- Each backend swapped in via config and verified against the mock-path baseline.
- **DoD:** `full_dub` produces a real dubbed video from a real Chinese clip.

### Stage 8 — `service` + `lipsync` stub + polish
- `service/`: FastAPI app exposing recipes over the LAN (submit job / poll status /
  fetch result; background execution) — for a future thin Mac client.
- `lipsync/base.py` `LipSyncBackend` ABC + `lipsync/stub.py` no-op passthrough — a clean
  seam for later (Wav2Lip / LatentSync).
- Optional `translation/ollama.py` local backend.
- README, example recipe configs in `recipes/`, end-to-end docs.
- **DoD:** Mac client can drive a recipe on the box over SSH/LAN; docs complete.

---

## Cross-cutting concerns

- **CUDA-bound modules:** `separation`, `asr`, `tts`. `timing` is CPU but needs the
  `rubberband` binary. Marked in each module docstring.
- **Testing discipline:** `uv run pytest` is CUDA-free and green at every stage via
  mocks. GPU backends use `@pytest.mark.gpu`, auto-skipped without CUDA.
- **Secrets:** env / `.env` only, `SecretStr`. `.env` is git-ignored; `.env.example`
  documents the keys.
- **Commits:** committing happens only when the user asks. When asked, one commit per
  completed stage keeps history aligned with the Progress Tracker.

## Verification (per stage)

- Stages 0–6: `uv run pytest` + `uv run ruff check`, plus the stage's DoD command
  (e.g. Stage 6: run each recipe via the CLI on mocks and inspect outputs).
- Stage 7: `uv run pytest -m gpu` on the box, plus a real Chinese clip through
  `full_dub` and a human listen of the dubbed result.
- Stage 8: drive a recipe from the Mac against the box's FastAPI service.

---

## Progress Tracker

- [x] Stage 0 — Scaffold & tooling
- [x] Stage 1 — Core contracts (schemas, config, errors)
- [x] Stage 2 — media_io
- [x] Stage 3 — subtitle
- [x] Stage 4 — translation (DeepSeek + mock)
- [x] Stage 5 — Mock GPU backends
- [ ] Stage 6 — pipeline + cli + mixing
- [ ] Stage 7a — asr / faster-whisper
- [ ] Stage 7b — separation / demucs
- [ ] Stage 7c — tts / CosyVoice 2
- [ ] Stage 7d — timing / rubberband
- [ ] Stage 8 — service + lipsync stub + polish
