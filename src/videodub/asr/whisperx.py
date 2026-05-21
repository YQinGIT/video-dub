"""WhisperX ASR backend — CUDA-bound (Stage 7a).

WhisperX builds on faster-whisper and adds two things that matter for dubbing:
voice-activity detection, which chunks the audio so silent stretches cannot
provoke hallucinated text; and a wav2vec2 *forced-alignment* pass, which snaps
every word — and therefore every segment boundary — onto the waveform far more
precisely than Whisper's own attention-derived timestamps. Accurate boundaries
are what let a dubbed line start and end in sync, so this is the recommended
ASR backend for the full-dub recipe.

Heavy imports (`torch`, `whisperx`) happen at module top level; the
`videodub.asr` factory imports this module only when the backend is selected,
so `import videodub.asr` itself stays cheap and CUDA-free.

Diarization is deferred past Stage 7a, so `ASRConfig.diarize` is not acted on
here — WhisperX can integrate pyannote diarization when that work is picked up.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch  # noqa: F401 -- loads the CUDA runtime libs CTranslate2 needs
import whisperx

from videodub.asr.base import ASRBackend
from videodub.config import ASRConfig
from videodub.errors import BackendError
from videodub.schemas import Segment, Transcript

# WhisperX transcribes VAD-detected chunks in parallel; 16 is the upstream
# default and leaves large-v3 comfortable headroom in the RTX 5080's 16 GB.
_BATCH_SIZE = 16


def _to_segments(raw: list[dict]) -> list[Segment]:
    """Map WhisperX result dicts onto `Segment`s, dropping unusable entries.

    A dict with no text, or with missing / non-finite / disordered timestamps
    (which alignment can produce for a chunk that has no recognisable words), is
    skipped rather than allowed to break the `Transcript` contract.
    """
    segments: list[Segment] = []
    for item in raw:
        text = str(item.get("text", "")).strip()
        start = item.get("start")
        end = item.get("end")
        if not text or start is None or end is None:
            continue
        start, end = float(start), float(end)
        if not (math.isfinite(start) and math.isfinite(end)) or end < start:
            continue
        segments.append(Segment(start=start, end=end, text=text))
    return segments


class WhisperXASR(ASRBackend):
    """ASR via WhisperX — faster-whisper transcription plus forced alignment.

    The Whisper model and the per-language alignment models are loaded lazily on
    first use and cached on the instance: loading pulls weights into VRAM, so it
    must not happen merely because the backend was constructed. Reusing one
    instance loads each model only once.
    """

    def __init__(self) -> None:
        self._model = None
        self._model_key: tuple[str, str, str] | None = None
        self._align_cache: dict[str, tuple | None] = {}

    def _load_model(self, cfg: ASRConfig):
        """Return the cached WhisperX model, (re)loading it if the config changed."""
        key = (cfg.model_size, cfg.device, cfg.compute_type)
        if self._model is not None and self._model_key == key:
            return self._model
        try:
            model = whisperx.load_model(
                cfg.model_size,
                device=cfg.device,
                compute_type=cfg.compute_type,
                language=cfg.language,  # None -> autodetect
            )
        except Exception as exc:  # download, CUDA, cuDNN, or out-of-memory failure
            raise BackendError(
                f"could not load WhisperX model {cfg.model_size!r} on device "
                f"{cfg.device!r} ({cfg.compute_type}): {exc}"
            ) from exc
        self._model = model
        self._model_key = key
        return model

    def _load_align(self, language: str, device: str):
        """Return the cached `(model, metadata)` aligner for `language`, or None.

        WhisperX ships alignment models for a fixed set of languages; for any
        other language there is nothing to load and the caller keeps Whisper's
        own segment timestamps.
        """
        if language not in self._align_cache:
            try:
                self._align_cache[language] = whisperx.load_align_model(
                    language_code=language, device=device
                )
            except Exception:
                self._align_cache[language] = None
        return self._align_cache[language]

    def transcribe(self, audio: Path, cfg: ASRConfig) -> Transcript:
        audio = Path(audio)
        if not audio.exists():
            raise BackendError(f"audio file not found: {audio}")

        model = self._load_model(cfg)
        try:
            waveform = whisperx.load_audio(str(audio))
            result = model.transcribe(waveform, batch_size=_BATCH_SIZE)
        except Exception as exc:
            raise BackendError(f"WhisperX failed to transcribe {audio}: {exc}") from exc

        language = result.get("language") or cfg.language
        segments = result.get("segments", [])

        # Forced alignment: snap segment boundaries onto the waveform. Skipped
        # when there is nothing to align or no aligner exists for the language.
        aligner = self._load_align(language, cfg.device) if language else None
        if aligner is not None and segments:
            align_model, metadata = aligner
            try:
                aligned = whisperx.align(
                    segments,
                    align_model,
                    metadata,
                    waveform,
                    cfg.device,
                    return_char_alignments=False,
                )
            except Exception as exc:
                raise BackendError(
                    f"WhisperX alignment failed for {audio}: {exc}"
                ) from exc
            segments = aligned.get("segments", segments)

        return Transcript(segments=_to_segments(segments), language=language)
