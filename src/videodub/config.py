"""Pydantic configuration.

`Settings` is the single config object. Per-stage models select a backend by string,
so swapping mock <-> real backends is config-only — no code change. Secrets come from
the environment / `.env` as `SecretStr` and are never hard-coded.

Override anything via env vars with the `VIDEODUB_` prefix and `__` for nesting, e.g.
`VIDEODUB_ASR__BACKEND=mock` or `VIDEODUB_DEEPSEEK_API_KEY=sk-...`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ASRConfig(BaseModel):
    """Audio -> Transcript. CUDA-BOUND for the real backend."""

    backend: Literal["faster_whisper", "whisperx", "funasr", "mock"] = "faster_whisper"
    model_size: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str | None = None  # None -> autodetect
    diarize: bool = False


class TranslationConfig(BaseModel):
    """Transcript -> translated Transcript. Portable (DeepSeek API)."""

    backend: Literal["deepseek", "ollama", "mock"] = "deepseek"
    model: str = "deepseek-chat"
    source_language: str = "zh"
    target_language: str = "en"
    timing_aware: bool = True  # prompt the model to hit segment durations
    refine_source: bool = True  # first ask DeepSeek to fix ASR errors in the source


class SeparationConfig(BaseModel):
    """Vocal / background split. CUDA-BOUND for the real backend."""

    backend: Literal["demucs", "mock"] = "demucs"
    model: str = "htdemucs_ft"  # fine-tuned htdemucs — cleanest stems
    enabled: bool = True  # toggle the whole stage
    device: str = "cuda"


class TTSConfig(BaseModel):
    """Translated Transcript -> cloned speech. CUDA-BOUND for local backends."""

    backend: Literal["indextts2", "gpt_sovits", "elevenlabs", "mock"] = "indextts2"
    device: str = "cuda"
    reference_audio: Path | None = None  # None -> clone from source vocals


class TimingConfig(BaseModel):
    """Fit synth segments to target durations. Portable (rubberband binary)."""

    backend: Literal["rubberband", "mock"] = "rubberband"
    max_stretch: float = 1.3  # cap so the dub stays natural
    min_stretch: float = 0.7


class MixingConfig(BaseModel):
    """Dubbed vocals + preserved background -> final audio. Portable (ffmpeg)."""

    background_gain_db: float = 0.0
    vocal_gain_db: float = 0.0


class Settings(BaseSettings):
    """Top-level configuration, assembled from defaults, `.env`, and env vars."""

    model_config = SettingsConfigDict(
        env_prefix="VIDEODUB_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    work_dir: Path = Path("./workdir")
    keep_intermediates: bool = True

    # Secrets — from env / `.env` only.
    deepseek_api_key: SecretStr | None = None
    elevenlabs_api_key: SecretStr | None = None

    # Per-stage configuration.
    asr: ASRConfig = ASRConfig()
    translation: TranslationConfig = TranslationConfig()
    separation: SeparationConfig = SeparationConfig()
    tts: TTSConfig = TTSConfig()
    timing: TimingConfig = TimingConfig()
    mixing: MixingConfig = MixingConfig()
