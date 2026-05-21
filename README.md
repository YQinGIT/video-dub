# video-dub

**Turn a video in one language into a video in another — while keeping the original speaker's voice.**

`video-dub` is a local-first, modular toolkit for video localization. Give it a Chinese
video and it can hand you back an English-dubbed video that still *sounds like the same
person* — or just an English subtitle file, if that is all you need.

Primary use case: **Chinese video → English dub.** Other language pairs work too.

```
  input.mp4  ──►  [ separate ]──►[ transcribe ]──►[ translate ]──►[ speak ]──►[ time + mix ]  ──►  input.dubbed.mp4
   (Chinese)         voices         Chinese text     English text   English      synced audio        (English)
                   vs. music                                       voice clone
```

---

## Table of contents

- [Demo](#demo)
- [What it does](#what-it-does)
- [Requirements](#requirements)
- [Installation (from scratch)](#installation-from-scratch)
- [Quick start](#quick-start)
  - [Example A — Dub a Chinese video into English](#example-a--dub-a-chinese-video-into-english)
  - [Example B — Generate an English subtitle file for a Chinese video](#example-b--generate-an-english-subtitle-file-for-a-chinese-video)
  - [Example C — Use a different model for a Japanese video](#example-c--use-a-different-model-for-a-japanese-video)
  - [Try it with no GPU and no API key](#try-it-with-no-gpu-and-no-api-key)
- [Command reference](#command-reference)
- [Configuration](#configuration)
- [How it works (high-level design)](#how-it-works-high-level-design)

---

## Demo

A Chinese lecture clip dubbed into English by `video-dub`. The English translation is
spoken in a **clone of the original speaker's voice**, and the background audio is
preserved — only the speech is replaced.

<table>
<tr>
<th width="50%">▶️ Original — Chinese</th>
<th width="50%">▶️ Dubbed — English</th>
</tr>
<tr>
<td>

<video src="https://github.com/OWNER/REPO/raw/main/samples/luoxiang.mp4" controls muted width="100%"></video>

</td>
<td>

<video src="https://github.com/OWNER/REPO/raw/main/samples/luoxiang.dubbed.mp4" controls muted width="100%"></video>

</td>
</tr>
</table>

<!--
  Before pushing: replace OWNER/REPO in the two URLs above with your GitHub path
  (e.g. yiquanwen/video-dub), and replace `main` if your default branch differs.
  GitHub plays a committed video inline only through an absolute raw URL inside a
  <video> tag — a relative path renders as a download link instead.
-->

> **Heads-up:** the two videos above won't play until `OWNER/REPO` in their URLs is
> replaced with this repository's real GitHub path. The source files live in
> [`samples/`](samples/) and are tracked in git.

---

## What it does

`video-dub` is built around a **pipeline**: the work is split into independent *stages*,
and each stage hands a clean, typed result to the next. You don't run the stages by
hand — you pick a **recipe** (a named, pre-wired sequence of stages) and the tool runs
them for you.

A few terms used throughout this README, in plain English:

| Term | What it means |
|------|---------------|
| **ASR** (Automatic Speech Recognition) | Listening to audio and writing down the words — i.e. transcription. |
| **TTS** (Text-to-Speech) | The reverse: turning written text into spoken audio. |
| **Source separation** | Splitting a soundtrack into *vocals* (the speech) and *background* (music, sound effects), so the background can be kept untouched. |
| **Voice cloning** | Making the TTS voice sound like a specific person, by giving it a short sample of that person speaking. |
| **Diarization** | Labelling *who* spoke each line when there are multiple speakers. |
| **Backend** | One concrete implementation of a stage. Each stage has several interchangeable backends — a real one (e.g. a neural network) and a `mock` one (fast, fake, for testing). |
| **Recipe** | A named sequence of stages. You choose a recipe; the tool runs the stages. |

The three recipes you'll use most:

| Recipe | Command shortcut | Output |
|--------|------------------|--------|
| `full_dub` | `videodub dub` | A new video file, voiced in the target language — plus a matching `.srt` subtitle file. |
| `translate_subtitles` | `videodub subtitle` | A translated subtitle file (`.srt`). |
| `transcribe` | `videodub transcribe` | A subtitle file in the *original* language (no translation). |

There is also `refine_subtitles` (`videodub refine`), which proofreads an existing
subtitle file to clean up speech-recognition mistakes.

---

## Requirements

- **Python 3.12** — exactly 3.12 (not 3.11, not 3.13).
- **[`uv`](https://docs.astral.sh/uv/)** — the package manager this project uses.
- **`ffmpeg`** (with `ffprobe`) on your `PATH` — used to read and write video/audio.
- **`rubberband`** CLI — used to stretch synthesized speech so it stays in sync.
  Only needed for the **full dub** recipe.
- **An NVIDIA GPU with CUDA** — needed for the real speech-recognition, separation,
  and text-to-speech models. (See [Try it with no GPU](#try-it-with-no-gpu-and-no-api-key)
  if you don't have one.)
- **A [DeepSeek](https://platform.deepseek.com/) API key** — used for translation. This
  is a paid cloud API; translation is the one stage that is *not* fully local.

---

## Installation (from scratch)

These steps assume a clean machine with an NVIDIA GPU.

**1. Install `uv`** (the package manager):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Install the system tools** (`ffmpeg` and `rubberband`). On Debian/Ubuntu:

```bash
sudo apt update
sudo apt install -y ffmpeg rubberband-cli
```

**3. Get the code and install Python dependencies:**

```bash
git clone <this-repo-url> video-dub
cd video-dub

uv sync --extra gpu     # installs everything, including the CUDA model stack
```

> `uv sync` (without `--extra gpu`) installs only the lightweight, portable parts.
> Use that if you just want to try the [no-GPU mock path](#try-it-with-no-gpu-and-no-api-key).

**4. Add your DeepSeek API key.** Copy the example env file and fill it in:

```bash
cp .env.example .env
```

Then edit `.env` so it reads:

```dotenv
VIDEODUB_DEEPSEEK_API_KEY=sk-your-real-key-here
```

The `.env` file is git-ignored, so your key stays out of version control.

**5. (Full dub only) Install the text-to-speech model.** The dubbing voice is produced
by **IndexTTS-2**, which needs its own separate Python environment. Follow the setup
instructions in the module docstring at
[`src/videodub/tts/indextts2.py`](src/videodub/tts/indextts2.py). You can skip this step
if you only need subtitles.

**6. Check that it works:**

```bash
uv run pytest          # runs the test suite — works without a GPU
uv run videodub recipes  # lists the available recipes
```

> **Running commands:** every command below is written as `uv run videodub ...`.
> The `uv run` prefix makes sure the command runs inside the project's environment.
> If you'd rather type just `videodub`, activate the environment first with
> `source .venv/bin/activate`.

---

## Quick start

### Example A — Dub a Chinese video into English

This produces a brand-new video, spoken in English, in the original speaker's voice.
Chinese-to-English is the **default** direction, so no configuration is needed:

```bash
uv run videodub dub my_video.mp4
```

Output: **`my_video.dubbed.mp4`** next to the input, alongside a matching subtitle file
**`my_video.dubbed.srt`**. The two share a name, so most video players load the
subtitles automatically when the dubbed video is opened.

For Chinese speech specifically, we recommend adding the bundled
[`recipes/zh_dub.toml`](recipes/zh_dub.toml) config. It swaps in **FunASR**, a
transcription model trained almost entirely on Mandarin, which is noticeably more
accurate on Chinese audio than the general-purpose default:

```bash
uv run videodub dub my_video.mp4 --config recipes/zh_dub.toml
```

> **What this needs:** a GPU, the IndexTTS-2 voice model (step 5 above), `rubberband`,
> and a DeepSeek API key.

### Example B — Generate an English subtitle file for a Chinese video

If you only want subtitles — no new audio, no new video — use the `subtitle` command.
It transcribes the Chinese speech, translates it to English, and writes a subtitle file:

```bash
uv run videodub subtitle my_video.mp4
```

Output: **`my_video.translated.srt`** next to the input.

This is much lighter than a full dub: it does **not** need the TTS voice model or
`rubberband`. It still needs a GPU (for transcription) and a DeepSeek API key (for
translation). As in Example A, add `--config recipes/zh_dub.toml` for the more
accurate Chinese transcription model:

```bash
uv run videodub subtitle my_video.mp4 --config recipes/zh_dub.toml
```

To choose the output filename yourself:

```bash
uv run videodub subtitle my_video.mp4 --output english_subs.srt
```

### Example C — Use a different model for a Japanese video

The default transcription model is tuned for Chinese. For a **Japanese** video we need
to (1) pick a transcription model that handles Japanese well, and (2) tell the
translator the source language is now Japanese instead of Chinese.

Both changes go in a small config file. Create a file called `ja_subtitle.toml`:

```toml
# Japanese video -> English subtitles.

[asr]
backend  = "whisperx"   # a different model: multilingual, with accurate word timing
language = "ja"         # tell it the audio is Japanese

[translation]
source_language = "ja"  # translate FROM Japanese...
target_language = "en"  # ...TO English
```

Then run the `subtitle` command, pointing `--config` at that file:

```bash
uv run videodub subtitle japanese_video.mp4 --config ja_subtitle.toml
```

Output: **`japanese_video.translated.srt`**.

The available transcription backends are `faster_whisper` (the default),
`whisperx` (Whisper with more accurate word-level timing — recommended here),
`funasr` (Chinese only), and `mock`. The same `ja_subtitle.toml` file works with the
`dub` command too, if you want a full Japanese-to-English dub.

### Try it with no GPU and no API key

Every stage ships with a **mock** backend — a fast, fake stand-in that produces
placeholder output. The bundled [`recipes/mock.toml`](recipes/mock.toml) selects mock
backends everywhere, so the whole pipeline runs on any laptop:

```bash
uv sync                                                   # core install, no GPU stack
uv run videodub dub my_video.mp4 --config recipes/mock.toml
```

The result won't be a real dub, but it proves your installation works end to end. This
is also how the automated tests run.

---

## Command reference

The installed command is **`videodub`**. It has one general command, `run`, plus four
shortcuts for the common recipes.

```bash
videodub run <recipe> <input> [options]   # run any recipe by name
videodub recipes                          # list all recipes and their stages

videodub dub        <video>      # shortcut for: run full_dub
videodub subtitle   <video>      # shortcut for: run translate_subtitles
videodub transcribe <video>      # shortcut for: run transcribe
videodub refine     <subtitle>   # shortcut for: run refine_subtitles
```

**Options** (accepted by every command above):

| Option | Short | Description |
|--------|-------|-------------|
| `--config <file>` | `-c` | A `.toml` or `.json` file selecting backends and settings. |
| `--output <path>` | `-o` | Where to write the result. Defaults to a path next to the input. |

**Recipes and their default output names:**

| Recipe | Stages | Default output |
|--------|--------|----------------|
| `full_dub` | extract → separate → transcribe → refine → translate → speak → time → mix → remux → render subtitles | `<name>.dubbed.mp4` + `<name>.dubbed.srt` |
| `translate_subtitles` | extract → transcribe → refine → translate → render subtitles | `<name>.translated.srt` |
| `transcribe` | extract → transcribe → render subtitles | `<name>.srt` |
| `refine_subtitles` | load subtitles → proofread → render subtitles | overwrites the input file |

`videodub refine` takes a **subtitle file** (`.srt` or `.vtt`), not a video. It sends
the text to DeepSeek to fix speech-recognition errors and writes the cleaned-up file
back — useful for polishing subtitles in bulk without re-running the pipeline.

---

## Configuration

You almost never edit code. Instead you point the tool at different **backends** and
**settings** through configuration. Settings are resolved in this order, each layer
overriding the one below it:

1. A `--config` file (TOML or JSON), if you pass one.
2. Environment variables prefixed with `VIDEODUB_`.
3. A `.env` file in the current directory (this is where your API key lives).
4. Built-in defaults.

A config file has one `[section]` per stage. You only list the things you want to
change — everything else keeps its default. A fuller example:

```toml
[asr]
backend  = "whisperx"   # faster_whisper | whisperx | funasr | mock
language = "ja"         # null = auto-detect

[translation]
backend         = "deepseek"   # deepseek | mock
source_language = "ja"
target_language = "en"

[separation]
backend = "demucs"      # demucs | mock
enabled = true          # set false to skip the vocals/background split

[tts]
backend = "indextts2"   # indextts2 | mock

[timing]
backend = "rubberband"  # rubberband | mock
```

The same values can be set as environment variables instead. Nested settings use a
double underscore (`__`):

```bash
export VIDEODUB_ASR__BACKEND=whisperx
export VIDEODUB_TRANSLATION__SOURCE_LANGUAGE=ja
```

**Secrets** — your DeepSeek API key — are read only from the environment or `.env`,
never from a `--config` file:

```dotenv
VIDEODUB_DEEPSEEK_API_KEY=sk-...
```

---

## How it works (high-level design)

`video-dub` is designed as a **pipeline of independent stages**. The guiding ideas:

- **One stage, one job.** Each stage does a single, well-defined transformation and is
  a self-contained Python module.
- **Typed contracts between stages.** Stages communicate through
  [Pydantic](https://docs.pydantic.dev/) data objects (`Transcript`, `Segment`,
  `SeparatedAudio`, `SynthesizedAudio`). A stage doesn't care *how* the previous stage
  did its work — only that the data matches the contract. This is what makes backends
  interchangeable.
- **Recipes are just data.** A recipe is an ordered list of stage names. Adding a new
  workflow means adding a list, not writing orchestration code.
- **Backends are swappable, and chosen by config.** Every stage offers a real backend
  and a `mock` one. A factory function picks the backend named in your config and
  *lazily imports* it — so simply importing the project never loads heavy GPU
  libraries unless you actually use them. Swapping mock ⇄ real is a one-line config
  change, never a code change.
- **Local-first.** Everything runs on your own machine except translation, which calls
  the DeepSeek cloud API.

### The stages and the models behind them

| Stage | What it does | Model / tool used | Runs on |
|-------|--------------|-------------------|---------|
| **Media I/O** | Extracts audio from the video; later puts the new audio back in. | `ffmpeg` | CPU |
| **Separation** | Splits the soundtrack into *vocals* and *background* so music and sound effects survive into the dub. | **Demucs** (`htdemucs_ft`) | GPU |
| **ASR** (transcription) | Converts speech to timestamped text. | **faster-whisper** (`large-v3`, default), **WhisperX** (better word timing), or **FunASR** (`Paraformer-zh`, Mandarin specialist) | GPU |
| **Refine** | Optionally proofreads the transcript to fix recognition errors before translating. | **DeepSeek** (`deepseek-chat`) | Cloud API |
| **Translation** | Translates each segment, keeping the original timestamps. It is *timing-aware* — the prompt tells the model how long each line has, so the translation fits the available time. | **DeepSeek** (`deepseek-chat`) | Cloud API |
| **TTS** (speech synthesis) | Speaks the translated text, *cloning the original speaker's voice* from the separated vocals. | **IndexTTS-2** (zero-shot, cross-lingual voice cloning) | GPU |
| **Timing** | Stretches or compresses each synthesized clip so it lines up with the original timeline, without sounding unnatural. | **Rubberband** | CPU |
| **Subtitle** | Renders a `Transcript` to `.srt` / `.vtt` / `.ass` (or parses one back in). | built-in | CPU |
| **Mixing** | Combines the dubbed voice with the preserved background music. | `ffmpeg` (`amix`) | CPU |

A **full dub** runs the whole chain, and finishes by writing a subtitle file next to
the dubbed video — so the dub always ships with matching subtitles. A **subtitle** or
**transcribe** run uses only the left-hand side — extract, transcribe, (translate,)
render — which is why those recipes need no GPU-heavy TTS and no `rubberband`.

### Project layout

```
src/videodub/
  schemas.py      data contracts shared between stages
  config.py       all settings (Pydantic)
  media_io/       audio extraction, remuxing (ffmpeg)
  separation/     vocals vs. background  (demucs | mock)
  asr/            speech -> text         (faster_whisper | whisperx | funasr | mock)
  translation/    text -> translated     (deepseek | mock)
  tts/            text -> speech         (indextts2 | mock)
  timing/         fit speech to timeline (rubberband | mock)
  subtitle/       render / parse subtitles
  mixing/         voice + background mix
  pipeline/       recipes + the runner that executes them
  cli/            the `videodub` command
recipes/          ready-made config files (mock.toml, zh_dub.toml)
tests/            test suite (runs without a GPU)
```

---

## Project status

Under active, stage-by-stage construction. See **[PLAN.md](PLAN.md)** for the full build
plan and progress tracker.
