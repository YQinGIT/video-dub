"""faster-whisper ASR backend — CUDA-bound (Stage 7a).

Runs OpenAI's Whisper through CTranslate2 (the `faster-whisper` package), which
is markedly faster and lighter on VRAM than the reference implementation. The
model named by `cfg.model_size` (default `large-v3`) transcribes an audio file
into a timed `Transcript`.

`torch` and `faster_whisper` are imported at module top level — heavy imports
belong *here*, in the backend module. The `videodub.asr` factory imports this
module only when the faster-whisper backend is selected, so `import videodub.asr`
itself stays cheap and CUDA-free.

`torch` is imported purely for a side effect: its wheel bundles the cuBLAS and
cuDNN runtime libraries that CTranslate2's CUDA build needs, and importing it
makes them discoverable in-process. faster-whisper does not otherwise use torch.
"""

from __future__ import annotations

from pathlib import Path

import torch  # noqa: F401 -- loads the CUDA runtime libs CTranslate2 needs
from faster_whisper import WhisperModel

from videodub.asr.base import ASRBackend
from videodub.config import ASRConfig
from videodub.errors import BackendError
from videodub.schemas import Segment, Transcript


class FasterWhisperASR(ASRBackend):
    """ASR via faster-whisper / CTranslate2.

    The Whisper model is loaded lazily on the first `transcribe()` call, not in
    `__init__` — loading pulls several GB into VRAM, and merely *selecting* the
    backend (the factory, a config check) must not do that. The loaded model is
    cached on the instance and reloaded only if the model / device / precision
    config changes, so reusing one instance loads the model just once.
    """

    def __init__(self) -> None:
        self._model: WhisperModel | None = None
        self._model_key: tuple[str, str, str] | None = None

    def _load_model(self, cfg: ASRConfig) -> WhisperModel:
        """Return the cached Whisper model, (re)loading it if the config changed."""
        key = (cfg.model_size, cfg.device, cfg.compute_type)
        if self._model is not None and self._model_key == key:
            return self._model

        try:
            model = WhisperModel(
                cfg.model_size,
                device=cfg.device,
                compute_type=cfg.compute_type,
            )
        except Exception as exc:  # download, CUDA, cuDNN, or out-of-memory failure
            raise BackendError(
                f"could not load faster-whisper model {cfg.model_size!r} on "
                f"device {cfg.device!r} ({cfg.compute_type}): {exc}"
            ) from exc

        self._model = model
        self._model_key = key
        return model

    def transcribe(self, audio: Path, cfg: ASRConfig) -> Transcript:
        audio = Path(audio)
        if not audio.exists():
            raise BackendError(f"audio file not found: {audio}")

        model = self._load_model(cfg)
        try:
            # `language=None` lets Whisper autodetect. `word_timestamps` refines
            # the segment boundaries (better dub timing); `vad_filter` drops
            # non-speech so silent stretches do not provoke hallucinated text.
            segment_iter, info = model.transcribe(
                str(audio),
                language=cfg.language,
                word_timestamps=True,
                vad_filter=True,
            )
            # The iterator is lazy: consuming it is what actually runs inference.
            segments: list[Segment] = []
            for s in segment_iter:
                text = s.text.strip()
                if text:
                    segments.append(Segment(start=s.start, end=s.end, text=text))
        except Exception as exc:
            raise BackendError(
                f"faster-whisper failed to transcribe {audio}: {exc}"
            ) from exc

        return Transcript(segments=segments, language=info.language)
