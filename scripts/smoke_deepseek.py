"""Manual smoke test for the DeepSeek translation backend — makes a REAL call.

Run it yourself when you want to check the live API:

    uv run python scripts/smoke_deepseek.py

It is gated by the API key: with no DeepSeek key configured (env var or .env)
it prints a notice and exits without touching the network. The pytest suite
never imports this file, so `uv run pytest` stays offline and free.
"""

import sys

from videodub.config import Settings, TranslationConfig
from videodub.schemas import Segment, Transcript
from videodub.translation import get_translator


def main() -> int:
    settings = Settings()
    if settings.deepseek_api_key is None:
        print(
            "No DeepSeek API key configured — skipping the live smoke test.\n"
            "Set VIDEODUB_DEEPSEEK_API_KEY in your .env to run it."
        )
        return 0

    cfg = TranslationConfig(backend="deepseek", source_language="zh", target_language="en")
    source = Transcript(
        segments=[
            Segment(start=0.0, end=2.0, text="你好,欢迎回到我的频道。"),
            Segment(start=2.0, end=5.0, text="今天我们来测试一下翻译功能。"),
        ],
        language="zh",
    )

    print(
        f"Translating {len(source.segments)} segment(s) via DeepSeek "
        f"({cfg.model}, {cfg.source_language} -> {cfg.target_language})...\n"
    )
    result = get_translator(cfg, settings.deepseek_api_key).translate(source, cfg)

    for src, dst in zip(source.segments, result.segments, strict=True):
        print(f"  [{src.start:.1f}-{src.end:.1f}s]")
        print(f"    {src.text}")
        print(f"    {dst.text}")
    print("\nSmoke test OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
