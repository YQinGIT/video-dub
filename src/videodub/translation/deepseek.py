"""DeepSeek translation backend — calls the DeepSeek chat API over HTTPS.

DeepSeek exposes an OpenAI-compatible chat-completions endpoint. We send the
source segments in batches, ask (in JSON mode) for an object mapping each
segment's index to its translated text, and reassemble a `Transcript`.

The source subtitles come from an automatic speech-recognition (ASR) model, so
they can carry transcription errors — wrong homophones, dropped words, stray
punctuation. `refine()` is a separate operation that proofreads the source text
*in its own language*: it asks DeepSeek to repair recognition errors and leave
correct lines untouched. The pipeline runs it as its own stage before
translation, so the corrected transcript is what both the subtitle renderer and
the translation stage go on to see.

When `cfg.timing_aware` is set, the prompt also carries each segment's duration
and a rough character budget, so the model keeps lines short enough to be spoken
within their original time slot — otherwise a dub drifts steadily out of sync.

`httpx` is imported *here*, not in the package `__init__`, so importing
`videodub.translation` stays cheap until the DeepSeek backend is actually used.
"""

from __future__ import annotations

import json
import time

import httpx
from pydantic import SecretStr

from videodub.config import TranslationConfig
from videodub.errors import BackendError
from videodub.schemas import Segment, Transcript
from videodub.translation.base import (
    Translator,
    build_refinement,
    build_translation,
)

_API_URL = "https://api.deepseek.com/chat/completions"
_CHARS_PER_SECOND = 14.0  # rough spoken-English rate — only a hint to the model
_BATCH_SIZE = 20  # segments per request: keeps each prompt and JSON reply small
_TEMPERATURE = 0.3  # low — translation should be stable, not inventive
_TIMEOUT_S = 60.0
_MAX_RETRIES = 4
_BACKOFF_BASE_S = 1.0  # sleep between retries grows 1s, 2s, 4s, ...
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}  # rate-limit + transient server errors


def _char_budget(duration: float) -> int:
    """A loose cap on translated characters for a segment `duration` seconds long."""
    return max(1, round(duration * _CHARS_PER_SECOND))


# --------------------------------------------------------------------------- #
# Prompt construction — kept as standalone functions so it is unit-testable.   #
# --------------------------------------------------------------------------- #

def _system_prompt(cfg: TranslationConfig) -> str:
    """The system message: role, language pair, timing rule, output format."""
    parts = [
        "You are a professional subtitle translator preparing a video for "
        f"dubbing. Translate each numbered segment from '{cfg.source_language}' "
        f"to '{cfg.target_language}'. Preserve meaning, tone, and register; "
        "translate the segments as one connected passage, not in isolation.",
    ]
    if cfg.timing_aware:
        parts.append(
            "Each translation will be spoken aloud in place of the original "
            "audio, so it must fit its time slot: stay close to the stated "
            "character budget and never greatly exceed it. Prefer concise "
            "phrasing over a literal rendering when space is tight."
        )
    parts.append(
        'Respond with ONLY a JSON object of the form '
        '{"translations": {"0": "<text>", "1": "<text>"}} — exactly one entry '
        "per input segment, keyed by the segment number as a string."
    )
    return " ".join(parts)


def _user_prompt(batch: list[Segment], cfg: TranslationConfig) -> str:
    """The user message: every segment in the batch as its own labelled block."""
    blocks: list[str] = []
    for index, seg in enumerate(batch):
        if cfg.timing_aware:
            header = (
                f"Segment {index} | duration {seg.duration:.1f}s | "
                f"budget ~{_char_budget(seg.duration)} chars"
            )
        else:
            header = f"Segment {index}"
        blocks.append(f"{header}\n{seg.text}")
    return "\n\n".join(blocks)


def build_messages(batch: list[Segment], cfg: TranslationConfig) -> list[dict[str, str]]:
    """The full `messages` array for one translation request (system + user)."""
    return [
        {"role": "system", "content": _system_prompt(cfg)},
        {"role": "user", "content": _user_prompt(batch, cfg)},
    ]


# --------------------------------------------------------------------------- #
# Prompt construction — the ASR-correction (source-refinement) pass.           #
# This pass proofreads the source subtitles *before* they are translated.      #
# --------------------------------------------------------------------------- #

def _refine_system_prompt(cfg: TranslationConfig) -> str:
    """The system message for the correction pass: fix ASR errors, nothing else."""
    return (
        "You are a meticulous proofreader of video subtitles written in the "
        f"language '{cfg.source_language}'. These subtitles were produced by an "
        "automatic speech-recognition (ASR) model, so they may contain "
        "transcription errors: wrong homophones, missing or duplicated words, "
        "run-on segments, or misplaced punctuation. Read the numbered segments "
        "as one connected passage and repair any such errors so each segment "
        "reads the way the speaker most likely intended. If a segment already "
        "looks correct, return it exactly as given. Never translate, "
        "paraphrase, summarise, or add or drop content — only fix recognition "
        "errors, and keep every segment in its original language. "
        'Respond with ONLY a JSON object of the form '
        '{"corrections": {"0": "<text>", "1": "<text>"}} — exactly one entry '
        "per input segment, keyed by the segment number as a string."
    )


def _refine_user_prompt(batch: list[Segment]) -> str:
    """The user message for the correction pass: each segment as a labelled block."""
    blocks = [f"Segment {index}\n{seg.text}" for index, seg in enumerate(batch)]
    return "\n\n".join(blocks)


def build_refine_messages(
    batch: list[Segment], cfg: TranslationConfig
) -> list[dict[str, str]]:
    """The full `messages` array for one ASR-correction request (system + user)."""
    return [
        {"role": "system", "content": _refine_system_prompt(cfg)},
        {"role": "user", "content": _refine_user_prompt(batch)},
    ]


# --------------------------------------------------------------------------- #
# Response parsing                                                            #
# --------------------------------------------------------------------------- #

def _parse_response(
    data: dict,
    batch_size: int,
    *,
    key: str = "translations",
    label: str = "translation",
) -> list[str]:
    """Extract `batch_size` strings, in index order, from a chat reply.

    The chat message `content` is itself a JSON document (the API was asked for
    JSON mode); we parse it and read its `key` object, requiring one entry per
    segment in the batch. Both passes share this code: the translation pass
    reads `"translations"`, the correction pass reads `"corrections"` — `label`
    only changes the wording of the error messages.

    Raises `BackendError` for any shape the contract does not allow.
    """
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise BackendError(f"unexpected DeepSeek response shape: {data!r}") from exc

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise BackendError(
            f"DeepSeek returned non-JSON content: {content[:200]!r}"
        ) from exc

    entries = payload.get(key) if isinstance(payload, dict) else None
    if not isinstance(entries, dict):
        raise BackendError(f"DeepSeek JSON has no {key!r} object: {payload!r}")

    texts: list[str] = []
    for index in range(batch_size):
        entry_key = str(index)
        if entry_key not in entries:
            raise BackendError(f"DeepSeek omitted a {label} for segment {index}")
        texts.append(str(entries[entry_key]))
    return texts


# --------------------------------------------------------------------------- #
# The backend                                                                 #
# --------------------------------------------------------------------------- #

class DeepSeekTranslator(Translator):
    """Translation via the DeepSeek chat API.

    `transport` is an injection seam for tests: pass an `httpx.MockTransport` to
    drive the client with canned responses and no real network. Leave it `None`
    in production.
    """

    def __init__(
        self, api_key: SecretStr, transport: httpx.BaseTransport | None = None
    ) -> None:
        self._api_key = api_key
        self._transport = transport

    def translate(self, transcript: Transcript, cfg: TranslationConfig) -> Transcript:
        if not transcript.segments:
            return build_translation(transcript, [], cfg)

        texts: list[str] = []
        with httpx.Client(transport=self._transport, timeout=_TIMEOUT_S) as client:
            for start in range(0, len(transcript.segments), _BATCH_SIZE):
                batch = transcript.segments[start : start + _BATCH_SIZE]
                reply = self._post_batch(client, build_messages(batch, cfg), cfg)
                texts.extend(_parse_response(reply, len(batch)))
        return build_translation(transcript, texts, cfg)

    def refine(self, transcript: Transcript, cfg: TranslationConfig) -> Transcript:
        """Proofread the ASR transcript, returning a same-language corrected one.

        Batched and retried exactly like `translate`, but each request carries
        the correction prompt and the reply is a `"corrections"` object. The
        result keeps the source language and timestamps — only the text changes.
        """
        if not transcript.segments:
            return transcript

        texts: list[str] = []
        with httpx.Client(transport=self._transport, timeout=_TIMEOUT_S) as client:
            for start in range(0, len(transcript.segments), _BATCH_SIZE):
                batch = transcript.segments[start : start + _BATCH_SIZE]
                reply = self._post_batch(client, build_refine_messages(batch, cfg), cfg)
                texts.extend(
                    _parse_response(
                        reply, len(batch), key="corrections", label="correction"
                    )
                )
        return build_refinement(transcript, texts)

    def _post_batch(
        self,
        client: httpx.Client,
        messages: list[dict[str, str]],
        cfg: TranslationConfig,
    ) -> dict:
        """POST one prepared `messages` array, retrying transient failures.

        Retries network errors and rate-limit / 5xx responses with exponential
        backoff. A non-retryable HTTP error (e.g. 401 bad key) fails immediately.
        Both the correction and translation passes funnel through here.
        """
        payload = {
            "model": cfg.model,
            "messages": messages,
            "temperature": _TEMPERATURE,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = client.post(_API_URL, json=payload, headers=headers)
            except httpx.TransportError as exc:  # connect / read / timeout failure
                last_error = exc
            else:
                if response.status_code < 400:
                    return response.json()
                if response.status_code not in _RETRYABLE_STATUS:
                    raise BackendError(
                        f"DeepSeek API error HTTP {response.status_code}: "
                        f"{response.text[:200]}"
                    )
                last_error = BackendError(
                    f"DeepSeek API transient HTTP {response.status_code}"
                )

            if attempt < _MAX_RETRIES - 1:
                time.sleep(_BACKOFF_BASE_S * 2**attempt)

        raise BackendError(
            f"DeepSeek API unreachable after {_MAX_RETRIES} attempts"
        ) from last_error
