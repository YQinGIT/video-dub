"""Deterministic, offline ASR backend.

`MockASR` does not really transcribe — with no model loaded, it cannot. It
returns a fixed, hand-written Chinese `Transcript` regardless of what the audio
actually says, so the pipeline can run and be tested on any machine. The output
is identical on every call, which lets pipeline tests assert on exact text and
timing.

It still checks that the audio file exists: a mock that silently accepted a
missing path would hide pipeline-wiring bugs instead of catching them.
"""

from __future__ import annotations

from pathlib import Path

from videodub.asr.base import ASRBackend
from videodub.config import ASRConfig
from videodub.errors import BackendError
from videodub.schemas import Segment, Transcript

# A fixed three-line transcript — (start, end, text). Chinese, because the
# project's primary use case is a Chinese source video.
_MOCK_LINES: list[tuple[float, float, str]] = [
    (0.0, 2.4, "欢迎观看这个视频。"),
    (2.4, 5.1, "这是一段示例字幕。"),
    (5.1, 7.0, "谢谢大家观看。"),
]


class MockASR(ASRBackend):
    """A fake ASR backend: returns a constant Chinese transcript."""

    def transcribe(self, audio: Path, cfg: ASRConfig) -> Transcript:
        audio = Path(audio)
        if not audio.exists():
            raise BackendError(f"audio file not found: {audio}")

        # `cfg.language` is None for autodetect; the mock "detects" Chinese.
        language = cfg.language or "zh"
        segments = [
            Segment(start=start, end=end, text=text)
            for start, end, text in _MOCK_LINES
        ]
        return Transcript(segments=segments, language=language)
