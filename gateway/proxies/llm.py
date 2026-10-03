"""LLM upstream: any OpenAI-compatible base URL (Ollama, LiteLLM, ...) from the policy snapshot.

- The agent's headers are never forwarded: requests carry only the router key named by
  ``upstreams.llm.api_key_env``, if any.
- The upstream is always called with ``stream: false``. Post controls need the complete
  answer, and an SSE chunk that reached the agent cannot be retracted, so a streaming agent
  gets the approved answer re-emitted as ``chat.completion.chunk`` events (SPEC "Streaming").
- Responses are bounded by ``limits.max_response_bytes``; upstream failures surface as a
  generic `UpstreamError`, never with upstream text.
"""

import json
import logging
import os
import time
from collections.abc import Iterator, Mapping
from typing import Any, Final

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gateway.policy.loader import PolicySnapshot
from gateway.upstream import (
    TokenUsage,
    Upstream,
    UpstreamError,
    UpstreamResult,
    refuse_encoded,
    wire_chunks,
)

logger = logging.getLogger(__name__)

SSE_CONTENT_CHARS: Final = 64  # content characters per re-emitted chunk
_MESSAGE_CORE_FIELDS: Final = frozenset({"role", "content", "tool_calls"})


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)


class FunctionCall(_Lenient):
    name: str
    arguments: str = ""


class ToolCall(_Lenient):
    id: str = ""
    type: str = "function"
    function: FunctionCall


class AssistantMessage(_Lenient):
    role: str = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class Choice(_Lenient):
    index: int = 0
    message: AssistantMessage
    finish_reason: str | None = None


class Usage(_Lenient):
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class ChatCompletion(_Lenient):
    """The upstream's answer; validated so a malformed one is an upstream error, not a crash."""

    id: str = ""
    created: int = 0
    model: str = ""
    choices: list[Choice]
    usage: Usage | None = None


class ModelCard(_Lenient):
    id: str = Field(min_length=1)


class ModelList(_Lenient):
    data: list[ModelCard]


class LLMProxy(Upstream):
    """httpx client to the OpenAI-compatible upstream; one connection pool per process."""

    def __init__(
        self,
        *,
        env: Mapping[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._env = env if env is not None else os.environ
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        if self._client is None:
            # trust_env=False: no proxy or netrc settings from the environment.
            self._client = httpx.AsyncClient(
                transport=self._transport, follow_redirects=False, trust_env=False
            )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def execute(self, payload: object, snapshot: PolicySnapshot) -> UpstreamResult:
        if not isinstance(payload, dict):
            raise UpstreamError("upstream_invalid_request")
        body: dict[str, Any] = {
            key: value
            for key, value in payload.items()  # pyright: ignore[reportUnknownVariableType] -- JSON object
            if key != "stream_options"  # only valid with stream: true
        }
        body["stream"] = False
        started = time.perf_counter()
        data = await self._send("POST", "/chat/completions", snapshot, body=body)
        elapsed = time.perf_counter() - started
        try:
            completion = ChatCompletion.model_validate(data)
        except ValidationError:
            raise UpstreamError("upstream_invalid_response") from None
        usage = (
            TokenUsage(
                model=completion.model or str(body.get("model", "")),
                prompt_tokens=completion.usage.prompt_tokens,
                completion_tokens=completion.usage.completion_tokens,
                total_tokens=completion.usage.total_tokens,
            )
            if completion.usage is not None
            else None
        )
        return UpstreamResult(body=data, elapsed_s=elapsed, usage=usage)

    async def list_models(self, snapshot: PolicySnapshot) -> list[dict[str, Any]]:
        """The upstream's ``/models`` entries, as returned."""
        data = await self._send("GET", "/models", snapshot)
        try:
            models = ModelList.model_validate(data)
        except ValidationError:
            raise UpstreamError("upstream_invalid_response") from None
        return [card.model_dump(mode="json") for card in models.data]

    def _headers(self, snapshot: PolicySnapshot) -> dict[str, str]:
        # identity only: a compressed body would be inflated before the size cap could see it.
        headers = {"accept": "application/json", "accept-encoding": "identity"}
        key_env = snapshot.policy.upstreams.llm.api_key_env
        if key_env is not None:
            key = self._env.get(key_env)
            if not key:
                logger.error("upstream key variable %s is not set", key_env)
                raise UpstreamError("upstream_misconfigured")
            headers["authorization"] = f"Bearer {key}"
        return headers

    async def _send(
        self, method: str, path: str, snapshot: PolicySnapshot, *, body: object = None
    ) -> object:
        if self._client is None:
            msg = "LLMProxy.start() was not awaited"
            raise RuntimeError(msg)
        limits = snapshot.policy.limits
        url = snapshot.policy.upstreams.llm.base_url.rstrip("/") + path
        raw = bytearray()
        try:
            async with self._client.stream(
                method,
                url,
                json=body,
                headers=self._headers(snapshot),
                timeout=limits.upstream_timeout_s,
            ) as response:
                if not response.is_success:
                    logger.warning("upstream %s %s answered %d", method, path, response.status_code)
                    raise UpstreamError("upstream_error")
                refuse_encoded(response)
                async for chunk in wire_chunks(response):
                    raw += chunk
                    if len(raw) > limits.max_response_bytes:
                        raise UpstreamError("upstream_response_too_large")
        except httpx.TimeoutException:
            raise UpstreamError("upstream_timeout") from None
        except httpx.HTTPError as exc:
            logger.warning("upstream %s %s failed: %s", method, path, type(exc).__name__)
            raise UpstreamError("upstream_unreachable") from None
        try:
            return json.loads(raw)
        except ValueError:
            raise UpstreamError("upstream_invalid_response") from None


# ------------------------------------------------------------------- SSE re-emission


def completion_chunks(
    completion: dict[str, Any], *, include_usage: bool = False
) -> Iterator[dict[str, Any]]:
    """An approved chat completion as OpenAI ``chat.completion.chunk`` objects.

    Per choice: the role, any extra message fields (e.g. ``reasoning``), the content in
    pieces, each tool call whole, then the finish reason. Usage comes last when asked for.
    """
    parsed = ChatCompletion.model_validate(completion)
    base: dict[str, Any] = {
        "id": parsed.id,
        "object": "chat.completion.chunk",
        "created": parsed.created,
        "model": parsed.model,
    }

    def chunk(index: int, delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
        return {**base, "choices": [{"index": index, "delta": delta, "finish_reason": finish}]}

    for choice in parsed.choices:
        message = choice.message.model_dump(mode="json", exclude_none=True)
        yield chunk(choice.index, {"role": choice.message.role})
        if extras := {k: v for k, v in message.items() if k not in _MESSAGE_CORE_FIELDS}:
            yield chunk(choice.index, extras)
        content = choice.message.content or ""
        for start in range(0, len(content), SSE_CONTENT_CHARS):
            yield chunk(choice.index, {"content": content[start : start + SSE_CONTENT_CHARS]})
        for position, call in enumerate(choice.message.tool_calls or ()):
            tool_call = {"index": position, **call.model_dump(mode="json", exclude_none=True)}
            yield chunk(choice.index, {"tool_calls": [tool_call]})
        yield chunk(choice.index, {}, choice.finish_reason or "stop")
    if include_usage and parsed.usage is not None:
        yield {**base, "choices": [], "usage": parsed.usage.model_dump(mode="json")}


def sse_events(completion: dict[str, Any], *, include_usage: bool = False) -> Iterator[bytes]:
    """Server-sent events for ``completion``, ending with ``data: [DONE]``."""
    for item in completion_chunks(completion, include_usage=include_usage):
        yield f"data: {json.dumps(item, separators=(',', ':'), ensure_ascii=False)}\n\n".encode()
    yield b"data: [DONE]\n\n"
