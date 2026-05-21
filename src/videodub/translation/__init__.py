"""translation — Transcript to translated Transcript. PORTABLE (DeepSeek API).

Public API:
    get_translator(cfg, api_key=None) -> Translator   select a backend by config

`get_translator` imports the chosen backend lazily, so importing this package
never pulls in `httpx` unless the DeepSeek backend is actually used.
"""

from __future__ import annotations

from pydantic import SecretStr

from videodub.config import TranslationConfig
from videodub.errors import ConfigError
from videodub.translation.base import Translator

__all__ = ["Translator", "get_translator"]


def get_translator(
    cfg: TranslationConfig, api_key: SecretStr | None = None
) -> Translator:
    """Return the translation backend named by `cfg.backend`.

    `api_key` is consumed only by the DeepSeek backend — pass
    `Settings.deepseek_api_key`. The mock backend ignores it.

    Raises `ConfigError` if the backend is unknown, unconfigured, or not yet
    implemented.
    """
    if cfg.backend == "mock":
        from videodub.translation.mock import MockTranslator

        return MockTranslator()

    if cfg.backend == "deepseek":
        if api_key is None:
            raise ConfigError(
                "DeepSeek translation backend selected but no API key was "
                "provided; set VIDEODUB_DEEPSEEK_API_KEY in your .env"
            )
        from videodub.translation.deepseek import DeepSeekTranslator

        return DeepSeekTranslator(api_key)

    if cfg.backend == "ollama":
        raise ConfigError(
            "translation backend 'ollama' is not implemented yet "
            "(planned for Stage 8); use 'deepseek' or 'mock'"
        )

    raise ConfigError(f"unknown translation backend: {cfg.backend!r}")
