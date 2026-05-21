"""Demucs source-separation backend — CUDA-bound (Stage 7b).

Demucs splits a recording into four stems — drums, bass, other, vocals.
Dubbing needs only a two-way split, so this backend keeps the `vocals` stem as
the isolated speech and sums the other three into a single `background` track
(music, sound effects, room tone) — the same "two-stems" reduction the Demucs
command-line tool performs.

The default model is `htdemucs_ft`, the fine-tuned Hybrid Transformer Demucs:
it produces the cleanest stems, and being ~4x slower than plain `htdemucs` is
negligible because separation runs once per video. Clean vocals matter twice
over — the stem feeds ASR now and the TTS voice-cloning reference later.

`torch` and `demucs` are imported at module top level; heavy imports belong
*here*, in the backend module. The `videodub.separation` factory imports this
module only when the demucs backend is selected, so `import videodub.separation`
itself stays cheap and CUDA-free.
"""

from __future__ import annotations

import wave
from pathlib import Path

import torch
from demucs.apply import apply_model
from demucs.audio import AudioFile
from demucs.pretrained import get_model

from videodub.config import SeparationConfig
from videodub.errors import BackendError
from videodub.schemas import SeparatedAudio
from videodub.separation.base import Separator

# The stem name Demucs models use for isolated speech. Every Demucs model lists
# a source called exactly "vocals"; the remaining stems sum into the background.
_VOCALS = "vocals"


def _write_wav(wav: torch.Tensor, path: Path, samplerate: int) -> None:
    """Write a `(channels, samples)` float tensor as a 16-bit PCM WAV file.

    Demucs ships its own `save_audio`, but on torchaudio 2.8 that routes through
    a codec backend whose ffmpeg shared libraries are not always discoverable,
    so it fails to write. The output is plain PCM WAV, so the standard-library
    `wave` module writes it directly — no dependency, nothing to misconfigure.
    Floats are peak-normalised with 1% headroom before quantising, so a stem
    louder than full scale is rescaled rather than hard-clipped.
    """
    peak = float(wav.abs().max())
    wav = wav / max(1.01 * peak, 1.0)
    pcm = (wav.clamp(-1.0, 1.0) * 32767.0).to(torch.int16)
    # WAV stores frames interleaved: (channels, samples) -> (samples, channels).
    interleaved = pcm.t().contiguous().cpu().numpy()
    with wave.open(str(path), "wb") as out:
        out.setnchannels(interleaved.shape[1])
        out.setsampwidth(2)  # 16-bit
        out.setframerate(samplerate)
        out.writeframes(interleaved.tobytes())


class DemucsSeparator(Separator):
    """Source separation via Demucs.

    The model is loaded lazily on the first `separate()` call, not in
    `__init__` — loading pulls weights from disk (downloading them on first
    ever use), and merely *selecting* the backend must not do that. The loaded
    model is cached on the instance and reloaded only if `cfg.model` changes,
    so reusing one instance loads the model just once.
    """

    def __init__(self) -> None:
        self._model = None
        self._model_name: str | None = None

    def _load_model(self, cfg: SeparationConfig):
        """Return the cached Demucs model, (re)loading it if `cfg.model` changed.

        The model is held on the CPU; `apply_model` moves it to `cfg.device`
        for the duration of inference, matching how the Demucs CLI works.
        """
        if self._model is not None and self._model_name == cfg.model:
            return self._model
        try:
            model = get_model(cfg.model)
        except Exception as exc:  # unknown model name or download failure
            raise BackendError(
                f"could not load Demucs model {cfg.model!r}: {exc}"
            ) from exc
        model.cpu()
        model.eval()
        self._model = model
        self._model_name = cfg.model
        return model

    def separate(
        self, audio: Path, cfg: SeparationConfig, out_dir: Path
    ) -> SeparatedAudio:
        audio = Path(audio)
        if not audio.exists():
            raise BackendError(f"audio file not found: {audio}")

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        model = self._load_model(cfg)
        if _VOCALS not in model.sources:
            raise BackendError(
                f"Demucs model {cfg.model!r} has no {_VOCALS!r} stem "
                f"(stems: {list(model.sources)})"
            )

        try:
            # Decode to the rate / channel count the model expects. Demucs's
            # AudioFile shells out to ffmpeg, so resampling is handled for us.
            wav = AudioFile(audio).read(
                streams=0,
                samplerate=model.samplerate,
                channels=model.audio_channels,
            )
            # Standardise loudness before inference, then undo it afterwards —
            # the model was trained on normalised input. `ref` is the mono mix.
            ref = wav.mean(0)
            wav = (wav - ref.mean()) / ref.std()

            # `wav[None]` adds the batch dimension apply_model expects; `[0]`
            # drops it again. Result shape: (n_sources, channels, samples).
            sources = apply_model(
                model, wav[None], device=cfg.device, progress=False
            )[0]
            sources = sources * ref.std() + ref.mean()
        except Exception as exc:
            raise BackendError(f"Demucs failed to separate {audio}: {exc}") from exc

        # Two-stem reduction: the vocal stem is the isolated speech; every
        # other stem sums into one background track.
        vocals_idx = model.sources.index(_VOCALS)
        vocals = sources[vocals_idx]
        background = torch.zeros_like(vocals)
        for idx, stem in enumerate(sources):
            if idx != vocals_idx:
                background += stem

        vocals_path = out_dir / "vocals.wav"
        background_path = out_dir / "background.wav"
        try:
            _write_wav(vocals, vocals_path, model.samplerate)
            _write_wav(background, background_path, model.samplerate)
        except Exception as exc:
            raise BackendError(
                f"Demucs could not write stems to {out_dir}: {exc}"
            ) from exc

        return SeparatedAudio(vocals=vocals_path, background=background_path)
