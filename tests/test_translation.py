"""Stage 4 — translation backend tests.

The DeepSeek tests never touch the real network. `httpx.MockTransport` lets us
hand the client a plain Python function that plays the role of the API: it
receives the outgoing request and returns a canned `httpx.Response`. So these
tests exercise the genuine request-building, batching, retry, and parsing code
— only the socket is replaced.
"""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import SecretStr

from videodub.config import TranslationConfig
from videodub.errors import BackendError, ConfigError
from videodub.schemas import Segment, Transcript
from videodub.translation import Translator, get_translator
from videodub.translation.deepseek import (
    DeepSeekTranslator,
    build_messages,
    build_refine_messages,
)
from videodub.translation.mock import MockTranslator

# A deepseek-backed config translating Chinese -> English, reused widely below.
ZH_EN = TranslationConfig(backend="deepseek", source_language="zh", target_language="en")


def _transcript(*texts: str) -> Transcript:
    """A zh Transcript with one 2-second segment per text, laid back to back."""
    segments = [
        Segment(start=float(i * 2), end=float(i * 2 + 2), text=t)
        for i, t in enumerate(texts)
    ]
    return Transcript(segments=segments, language="zh")


class FakeDeepSeek:
    """A stand-in DeepSeek endpoint for `httpx.MockTransport`.

    It serves both passes the backend can make. A *correction* request (its
    system prompt asks for a `"corrections"` object) gets each segment echoed
    back unchanged. A *translation* request gets `en-of-<source text>`, read
    back out of the prompt — so a test can verify the right source reached the
    right output slot, even across batches. `calls` counts every request.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        payload = json.loads(request.content)
        system_msg = payload["messages"][0]["content"]
        user_msg = payload["messages"][-1]["content"]
        refining = '"corrections"' in system_msg
        results: dict[str, str] = {}
        for index, block in enumerate(user_msg.split("\n\n")):
            source_text = block.split("\n", 1)[1]  # text after the "Segment N" header
            results[str(index)] = source_text if refining else f"en-of-{source_text}"
        key = "corrections" if refining else "translations"
        content = json.dumps({key: results})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def _translator(handler) -> DeepSeekTranslator:
    """A DeepSeekTranslator whose network is replaced by `handler`."""
    return DeepSeekTranslator(
        SecretStr("test-key"), transport=httpx.MockTransport(handler)
    )


# --------------------------------------------------------------------------- #
# Mock translator                                                             #
# --------------------------------------------------------------------------- #

def test_mock_translator_tags_target_language():
    cfg = TranslationConfig(backend="mock", target_language="en")
    out = MockTranslator().translate(_transcript("你好", "再见"), cfg)

    assert [s.text for s in out.segments] == ["[en] 你好", "[en] 再见"]
    assert out.language == "en"
    assert out.source_language == "zh"


def test_mock_translator_preserves_timestamps():
    src = _transcript("你好")
    out = MockTranslator().translate(src, TranslationConfig(backend="mock"))

    assert out.segments[0].start == src.segments[0].start
    assert out.segments[0].end == src.segments[0].end


def test_mock_translator_empty_transcript():
    out = MockTranslator().translate(Transcript(), TranslationConfig(backend="mock"))
    assert out.segments == []
    assert out.language == "en"


# --------------------------------------------------------------------------- #
# DeepSeek backend — happy path                                               #
# --------------------------------------------------------------------------- #

def test_deepseek_translates_and_preserves_timing():
    out = _translator(FakeDeepSeek()).translate(_transcript("你好", "再见"), ZH_EN)

    assert [s.text for s in out.segments] == ["en-of-你好", "en-of-再见"]
    # timestamps are carried over from the source untouched
    assert [(s.start, s.end) for s in out.segments] == [(0.0, 2.0), (2.0, 4.0)]
    assert out.language == "en"
    assert out.source_language == "zh"


def test_deepseek_batches_large_transcripts():
    # 25 segments > the batch size of 20, so this must take two requests.
    src = _transcript(*[f"zh-{i}" for i in range(25)])
    fake = FakeDeepSeek()
    out = _translator(fake).translate(src, ZH_EN)

    assert fake.calls == 2
    assert len(out.segments) == 25
    # segment 20 lands in the second batch — verify it kept its own source text
    assert out.segments[20].text == "en-of-zh-20"
    assert out.segments[24].text == "en-of-zh-24"


def test_deepseek_empty_transcript_makes_no_request():
    fake = FakeDeepSeek()
    out = _translator(fake).translate(Transcript(), ZH_EN)

    assert fake.calls == 0
    assert out.segments == []
    assert out.language == "en"


def test_deepseek_sends_bearer_token():
    seen: dict[str, str] = {}

    def capture(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization", "")
        return FakeDeepSeek()(request)

    _translator(capture).translate(_transcript("你好"), ZH_EN)
    assert seen["auth"] == "Bearer test-key"


# --------------------------------------------------------------------------- #
# DeepSeek backend — retry and error handling                                 #
# --------------------------------------------------------------------------- #

def test_deepseek_retries_transient_errors_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)  # do not actually wait
    attempts = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(503, text="service unavailable")
        return FakeDeepSeek()(request)

    out = _translator(flaky).translate(_transcript("你好"), ZH_EN)
    assert attempts["n"] == 3
    assert out.segments[0].text == "en-of-你好"


def test_deepseek_gives_up_after_persistent_errors(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)

    def always_500(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(BackendError, match="after 4 attempts"):
        _translator(always_500).translate(_transcript("你好"), ZH_EN)


def test_deepseek_does_not_retry_a_client_error():
    seen = {"n": 0}

    def unauthorized(_request: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        return httpx.Response(401, text="invalid api key")

    with pytest.raises(BackendError, match="HTTP 401"):
        _translator(unauthorized).translate(_transcript("你好"), ZH_EN)
    assert seen["n"] == 1  # 401 is not retryable — fail fast, no backoff


def test_deepseek_rejects_unparseable_json():
    def bad_json(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "not json at all"}}]}
        )

    with pytest.raises(BackendError, match="non-JSON"):
        _translator(bad_json).translate(_transcript("你好"), ZH_EN)


def test_deepseek_rejects_a_missing_segment():
    def incomplete(_request: httpx.Request) -> httpx.Response:
        # promises two segments but returns only one translation
        content = json.dumps({"translations": {"0": "hi"}})
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}]}
        )

    with pytest.raises(BackendError, match="omitted a translation"):
        _translator(incomplete).translate(_transcript("你好", "再见"), ZH_EN)


# --------------------------------------------------------------------------- #
# Prompt construction                                                         #
# --------------------------------------------------------------------------- #

def test_timing_aware_prompt_carries_duration_and_budget():
    batch = [Segment(start=0.0, end=3.0, text="你好")]
    messages = build_messages(batch, TranslationConfig(timing_aware=True))
    system, user = messages[0]["content"], messages[1]["content"]

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "duration 3.0s" in user
    assert "budget ~33 chars" in user  # 3.0 s * 11 chars/s
    assert "你好" in user
    assert "character budget" in system  # the timing instruction is present
    assert "translations" in system  # the JSON output contract is stated


def test_non_timing_prompt_omits_budget():
    batch = [Segment(start=0.0, end=3.0, text="你好")]
    messages = build_messages(batch, TranslationConfig(timing_aware=False))
    system, user = messages[0]["content"], messages[1]["content"]

    assert "budget" not in user
    assert "duration" not in user
    assert "Segment 0" in user
    assert "character budget" not in system


def test_prompt_names_the_language_pair():
    messages = build_messages([Segment(start=0.0, end=1.0, text="你好")], ZH_EN)
    system = messages[0]["content"]
    assert "'zh'" in system
    assert "'en'" in system


# --------------------------------------------------------------------------- #
# Source refinement — the ASR-correction pass                                 #
# --------------------------------------------------------------------------- #

def test_refine_prompt_describes_the_asr_correction_task():
    batch = [Segment(start=0.0, end=2.0, text="在见")]
    messages = build_refine_messages(batch, ZH_EN)
    system, user = messages[0]["content"], messages[1]["content"]

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    # the prompt explains where the text came from and what to do with it
    assert "speech-recognition" in system
    assert "'zh'" in system  # the source language is named
    assert '"corrections"' in system  # the JSON output contract is stated
    # the model is told to translate nothing in this pass
    assert "Never translate" in system
    # the source text is handed over under a plain "Segment N" header
    assert "Segment 0" in user
    assert "在见" in user


def test_deepseek_refine_corrects_a_segment():
    """`refine` repairs an ASR mistake while keeping the source language."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        user_msg = payload["messages"][-1]["content"]
        corrections: dict[str, str] = {}
        for index, block in enumerate(user_msg.split("\n\n")):
            text = block.split("\n", 1)[1]
            corrections[str(index)] = text.replace("在见", "再见")  # fix a homophone
        content = json.dumps({"corrections": corrections})
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}]}
        )

    out = _translator(handler).refine(_transcript("你好", "在见"), ZH_EN)

    assert [s.text for s in out.segments] == ["你好", "再见"]  # one fixed, one kept
    assert out.language == "zh"  # still the source language — not translated


def test_deepseek_refine_preserves_timestamps():
    src = _transcript("在见")
    out = _translator(FakeDeepSeek()).refine(src, ZH_EN)

    assert (out.segments[0].start, out.segments[0].end) == (0.0, 2.0)


def test_deepseek_refine_uses_the_correction_prompt():
    """`refine` must send the ASR-correction prompt, not the translation one."""
    seen: dict[str, str] = {}

    def capture(request: httpx.Request) -> httpx.Response:
        seen["system"] = json.loads(request.content)["messages"][0]["content"]
        return FakeDeepSeek()(request)

    _translator(capture).refine(_transcript("你好"), ZH_EN)
    assert "speech-recognition" in seen["system"]
    assert '"corrections"' in seen["system"]


def test_deepseek_refine_empty_transcript_makes_no_request():
    fake = FakeDeepSeek()
    out = _translator(fake).refine(Transcript(), ZH_EN)

    assert fake.calls == 0  # nothing to correct
    assert out.segments == []


def test_deepseek_refine_rejects_a_missing_segment():
    def incomplete(_request: httpx.Request) -> httpx.Response:
        # promises two segments but returns only one correction
        content = json.dumps({"corrections": {"0": "你好"}})
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}]}
        )

    with pytest.raises(BackendError, match="omitted a correction"):
        _translator(incomplete).refine(_transcript("你好", "再见"), ZH_EN)


def test_mock_translator_refine_is_a_pass_through():
    """The mock cannot proofread — `refine` returns the transcript untouched."""
    src = _transcript("你好", "再见")
    out = MockTranslator().refine(src, TranslationConfig(backend="mock"))

    assert out is src


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #

def test_factory_returns_mock_backend():
    backend = get_translator(TranslationConfig(backend="mock"))
    assert isinstance(backend, MockTranslator)
    assert isinstance(backend, Translator)


def test_factory_returns_deepseek_backend_with_key():
    backend = get_translator(TranslationConfig(backend="deepseek"), SecretStr("k"))
    assert isinstance(backend, DeepSeekTranslator)


def test_factory_deepseek_without_key_raises():
    with pytest.raises(ConfigError, match="no API key"):
        get_translator(TranslationConfig(backend="deepseek"))


def test_factory_ollama_not_implemented():
    with pytest.raises(ConfigError, match="not implemented"):
        get_translator(TranslationConfig(backend="ollama"))
