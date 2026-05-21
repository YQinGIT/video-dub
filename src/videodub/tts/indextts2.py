"""IndexTTS-2 TTS backend — runs IndexTTS-2 in its own isolated venv (Stage 7c).

IndexTTS-2 is a zero-shot cross-lingual voice-cloning model and the recommended
dub voice. Its dependencies are exact-pinned (an older numpy, transformers, and
numba, on Python <=3.11) and cannot share the videodub virtual environment
without breaking the Stage 7a WhisperX backend. So IndexTTS-2 is installed in a
*separate* venv, and this backend drives it out of process: it writes a JSON
job, runs `_indextts2_worker.py` with that venv's Python, and reads back the
WAV clips the worker produced.

Consequences of the subprocess design:
  * This module imports only the standard library and `videodub` — never torch
    or `indextts` — so `import videodub.tts.indextts2` stays cheap and
    CUDA-free, and the factory can build this backend without any GPU library.
  * The worker process loads the model once, synthesizes every segment, then
    exits, which fully releases the VRAM it used.

Install (its own venv — IndexTTS-2 cannot share the videodub venv):
    git clone https://github.com/index-tts/index-tts.git
    cd index-tts && uv sync --python 3.11
    hf download IndexTeam/IndexTTS-2 --local-dir checkpoints
Point `VIDEODUB_INDEXTTS_HOME` at that directory, or place it as `index-tts`
beside the videodub repo (the default location this backend looks in).
"""

from __future__ import annotations

import json
import os
import subprocess
import wave
from pathlib import Path

from videodub.config import TTSConfig
from videodub.errors import BackendError
from videodub.schemas import SynthesizedAudio, SynthSegment, Transcript
from videodub.tts.base import TTSBackend

# Environment variable that overrides where IndexTTS-2 is installed.
_ENV_HOME = "VIDEODUB_INDEXTTS_HOME"
# Default: an `index-tts` directory beside the videodub repository root.
# __file__ = <repo>/src/videodub/tts/indextts2.py -> parents[3] = <repo>.
_DEFAULT_HOME = Path(__file__).resolve().parents[3].parent / "index-tts"
# This package's synthesis worker, run with the IndexTTS-2 venv's interpreter.
_WORKER = Path(__file__).with_name("_indextts2_worker.py")
# Used only for the degenerate empty-transcript case; for any real clip the
# sample rate is read straight from the WAV the worker writes.
_OUTPUT_SAMPLE_RATE = 22050


def _resolve_home() -> Path:
    """Return the IndexTTS-2 install dir (`VIDEODUB_INDEXTTS_HOME` or default)."""
    override = os.environ.get(_ENV_HOME)
    return Path(override).expanduser() if override else _DEFAULT_HOME


def _wav_sample_rate(path: Path) -> int:
    """Return the sample rate recorded in a WAV file's header."""
    with wave.open(str(path), "rb") as wav:
        return wav.getframerate()


class IndexTTS2TTS(TTSBackend):
    """TTS via IndexTTS-2, driven out of process in its own isolated venv.

    There is nothing to cache on the instance: each `synthesize()` call spawns
    one worker process that loads the model, renders every segment, and exits.
    """

    def synthesize(
        self,
        transcript: Transcript,
        cfg: TTSConfig,
        out_dir: Path,
        reference_audio: Path | None = None,
    ) -> SynthesizedAudio:
        # Resolve to an absolute path before anything else: the worker runs as
        # a subprocess with its cwd set to the IndexTTS-2 repo, so the job file
        # and every clip path handed to it must be absolute — a path relative
        # to the caller's working directory would not resolve on the far side.
        out_dir = Path(out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        # Nothing to synthesize -> return early, before spawning anything.
        if not transcript.segments:
            return SynthesizedAudio(segments=[], sample_rate=_OUTPUT_SAMPLE_RATE)

        if reference_audio is None:
            raise BackendError(
                "IndexTTS-2 clones the speaker's voice and needs a reference "
                "clip; none was given (set TTSConfig.reference_audio, or run "
                "the separation stage so the isolated vocals can be used)."
            )
        reference_audio = Path(reference_audio)
        if not reference_audio.exists():
            raise BackendError(f"reference audio not found: {reference_audio}")

        home = _resolve_home()
        venv_python = home / ".venv" / "bin" / "python"
        model_dir = home / "checkpoints"
        cfg_path = model_dir / "config.yaml"
        if not venv_python.exists() or not cfg_path.exists():
            raise BackendError(
                f"IndexTTS-2 is not installed at {home} (looked for "
                f"{venv_python} and {cfg_path}). Install it in its own venv — "
                "see the install notes in videodub/tts/indextts2.py — or set "
                f"{_ENV_HOME} to its directory."
            )

        # Job file: absolute paths only, so the worker's working directory
        # cannot affect what it reads or writes.
        segments_spec: list[dict[str, str]] = []
        clip_paths: list[Path] = []
        for index, seg in enumerate(transcript.segments):
            clip = out_dir / f"segment_{index:04d}.wav"
            clip_paths.append(clip)
            segments_spec.append({"text": seg.text, "output_path": str(clip)})

        job_path = out_dir / "_indextts2_job.json"
        job_path.write_text(
            json.dumps(
                {
                    "cfg_path": str(cfg_path),
                    "model_dir": str(model_dir),
                    "device": cfg.device,
                    "reference_audio": str(reference_audio.resolve()),
                    "segments": segments_spec,
                }
            ),
            encoding="utf-8",
        )

        # Run the worker with IndexTTS-2's own interpreter. `cwd=home` lets any
        # path IndexTTS-2 resolves relative to its repo resolve correctly.
        try:
            proc = subprocess.run(
                [str(venv_python), str(_WORKER), str(job_path)],
                cwd=str(home),
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            raise BackendError(
                f"could not start the IndexTTS-2 worker: {exc}"
            ) from exc

        if proc.returncode != 0:
            raise BackendError(
                "IndexTTS-2 synthesis failed "
                f"(worker exit code {proc.returncode}):\n{proc.stderr.strip()}"
            )

        segments: list[SynthSegment] = []
        sample_rate = _OUTPUT_SAMPLE_RATE
        for seg, clip in zip(transcript.segments, clip_paths, strict=True):
            if not clip.exists():
                raise BackendError(
                    f"IndexTTS-2 worker reported success but {clip} is missing"
                )
            sample_rate = _wav_sample_rate(clip)
            segments.append(
                SynthSegment(
                    start=seg.start,
                    end=seg.end,
                    audio_path=clip,
                    text=seg.text,
                    speaker=seg.speaker,
                )
            )

        return SynthesizedAudio(segments=segments, sample_rate=sample_rate)
