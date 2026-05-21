"""FunASR / Paraformer-zh ASR backend — CUDA-bound.

Paraformer is a Mandarin-specialist speech recognizer from Alibaba's FunASR
toolkit. Unlike multilingual Whisper it is trained almost entirely on Chinese,
so it is more accurate on Mandarin — it hears proper nouns and domain terms
(e.g. 中国政法大学) that a Whisper backend tends to garble.

Segmentation. FunASR's one-call `AutoModel` pipeline merges its VAD chunks into
a single block of text, and its `sentence_timestamp` option crashes in the
installed release — so neither route yields the per-segment timing the dub
needs. This backend therefore drives the three FunASR models itself:

  1. `fsmn-vad` finds the speech spans (their start / end times);
  2. each span's audio is recognized by `paraformer-zh`;
  3. `ct-punc` restores punctuation so the line reads naturally.

Each VAD span becomes one timed `Segment`. A clip with no pause long enough to
split comes back as a single segment — which is fine: the translator simply
gets more context and the dub is one continuous line.

Weights download on first use from the HuggingFace mirror (`hub="hf"`) rather
than FunASR's default ModelScope hub, which is unreliable from outside China;
they are cached after the first run.

`funasr` and `librosa` are imported at module top level — heavy imports belong
*here*. The `videodub.asr` factory imports this module only when the funasr
backend is selected, so `import videodub.asr` itself stays cheap and CUDA-free.
"""

from __future__ import annotations

from pathlib import Path

import librosa
from funasr import AutoModel

from videodub.asr.base import ASRBackend
from videodub.config import ASRConfig
from videodub.errors import BackendError
from videodub.schemas import Segment, Transcript

# Paraformer-zh is Mandarin-only, so the transcript language is always zh;
# `ASRConfig.language` exists for the autodetecting Whisper backends. The
# Whisper-specific `model_size` / `compute_type` fields do not apply here.
_LANGUAGE = "zh"
# FunASR model names: voice-activity detection, the recognizer, punctuation.
_VAD_MODEL = "fsmn-vad"
_ASR_MODEL = "paraformer-zh"
_PUNC_MODEL = "ct-punc"
# fsmn-vad and Paraformer operate on 16 kHz mono audio; VAD spans come back as
# [start_ms, end_ms] pairs in milliseconds.
_SAMPLE_RATE = 16000
_MS_PER_S = 1000.0


class FunASRASR(ASRBackend):
    """ASR via FunASR / Paraformer-zh — a Mandarin-specialist recognizer.

    The three FunASR models (VAD, recognizer, punctuation) are loaded lazily on
    the first `transcribe()` call and cached on the instance: loading pulls
    weights into VRAM, so it must not happen merely because the backend was
    selected. Reusing one instance loads them just once.
    """

    def __init__(self) -> None:
        self._models: tuple[AutoModel, AutoModel, AutoModel] | None = None
        self._device: str | None = None

    def _load_models(self, cfg: ASRConfig) -> tuple[AutoModel, AutoModel, AutoModel]:
        """Return the cached (vad, asr, punc) models, reloading on device change."""
        if self._models is not None and self._device == cfg.device:
            return self._models
        try:
            vad = AutoModel(
                model=_VAD_MODEL, device=cfg.device, hub="hf", disable_update=True
            )
            asr = AutoModel(
                model=_ASR_MODEL, device=cfg.device, hub="hf", disable_update=True
            )
            punc = AutoModel(
                model=_PUNC_MODEL, device=cfg.device, hub="hf", disable_update=True
            )
        except Exception as exc:  # download, CUDA, or out-of-memory failure
            raise BackendError(
                f"could not load FunASR models on device {cfg.device!r}: {exc}"
            ) from exc
        self._models = (vad, asr, punc)
        self._device = cfg.device
        return self._models

    def transcribe(self, audio: Path, cfg: ASRConfig) -> Transcript:
        audio = Path(audio)
        if not audio.exists():
            raise BackendError(f"audio file not found: {audio}")

        vad, asr, punc = self._load_models(cfg)
        try:
            # 16 kHz mono is what fsmn-vad and Paraformer expect; librosa
            # resamples whatever the upstream stage produced (the Demucs vocal
            # stem is 44.1 kHz).
            waveform, _ = librosa.load(str(audio), sr=_SAMPLE_RATE, mono=True)
            # VAD returns speech spans as [[start_ms, end_ms], ...].
            spans = vad.generate(input=str(audio))[0].get("value") or []

            segments: list[Segment] = []
            for start_ms, end_ms in spans:
                a = int(start_ms / _MS_PER_S * _SAMPLE_RATE)
                b = int(end_ms / _MS_PER_S * _SAMPLE_RATE)
                clip = waveform[a:b]
                if clip.size == 0:
                    continue
                # Recognize this span, then restore its punctuation.
                recognized = asr.generate(input=clip)[0].get("text", "").strip()
                if not recognized:
                    continue
                punctuated = punc.generate(input=recognized)[0].get("text", "").strip()
                segments.append(
                    Segment(
                        start=start_ms / _MS_PER_S,
                        end=end_ms / _MS_PER_S,
                        text=punctuated or recognized,
                    )
                )
        except Exception as exc:
            raise BackendError(f"FunASR failed to transcribe {audio}: {exc}") from exc

        return Transcript(segments=segments, language=_LANGUAGE)
