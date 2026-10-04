"""LLM upstream: any OpenAI-compatible base URL (Ollama, LiteLLM, ...) from the policy snapshot.

- Two upstreams may be declared: ``upstreams.llm`` (local, the default) and
  ``upstreams.llm_remote`` (an opt-in remote router such as OpenRouter). The process selects
  one with ``ACL_LLM_UPSTREAM``; there is no fallback from one to the other.
- Inside the gateway models keep their logical ids. For the remote upstream the request's
  model is mapped to the provider id on the way out (an unmapped model is refused,
  ``model_not_mapped``) and the answer's model back to the logical id on the way in, so
  ``model_allowlist`` compares logical ids. Only OpenAI chat-completion fields are forwarded
  to it (router extensions such as ``models``, ``provider`` or ``plugins`` from the agent are
  dropped), then the policy's ``extra_body`` is merged in and wins.
- ``reasoning_effort`` is translated per upstream (``reasoning: openrouter``): ``none`` becomes
  ``reasoning: {"enabled": false}``, any other effort ``reasoning: {"effort": ...}``.
- The agent's headers are never forwarded: requests carry only the router key named by the
  selected upstream's ``api_key_env``, if any.
- The upstream is always called with ``stream: false``. Post controls need the complete
  answer, and an SSE chunk that reached the agent cannot be retracted, so a streaming agent
  gets the approved answer re-emitted as ``chat.completion.chunk`` events (SPEC "Streaming").
- Responses are bounded by ``limits.max_response_bytes``; upstream failures surface as a
  generic `UpstreamError`, never with upstream text.
"""

import copy
import json
import logging
import math
import os
import re
import time
from collections.abc import Callable, Iterator, Mapping
from typing import Any, Final, cast
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gateway.core.types import LlmUpstreamKind
from gateway.policy.loader import PolicySnapshot
from gateway.policy.schema import AnyLlmUpstream, Policy, RemoteLlmUpstream
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
_PRINTABLE_ID: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,63}")
# OpenAI chat-completion request fields a remote upstream receives from the agent's payload.
# Everything else is dropped: a router's own extensions (OpenRouter's `models` fallbacks,
# `provider` routing, `plugins` such as web search, `transforms`, `route`) would let the agent
# pick another model, relax data-retention routing or open egress the policy never granted.
REMOTE_FORWARDED_FIELDS: Final = frozenset(
    {
        "model",
        "messages",
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "stop",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "response_format",
        "reasoning_effort",
    }
)


class ModelNotMappedError(UpstreamError):
    """The remote upstream has no provider model for this logical id: refused before egress.

    The message names the models it does serve (operator configuration, no payload)."""

    status_code = 400

    def __init__(self, upstream: RemoteLlmUpstream, requested: object) -> None:
        super().__init__("model_not_mapped")
        served = ", ".join(upstream.model_map)
        model = (
            requested if isinstance(requested, str) and _PRINTABLE_ID.fullmatch(requested) else None
        )
        subject = f"{model} is local-only" if model else "this model is not served remotely"
        self.message = f"{subject}; the remote upstream serves: {served}"


class RemoteOnlyModelError(UpstreamError):
    """A remote logical id requested while the local upstream is selected: refused, since the
    local upstream would only answer an unhelpful 404."""

    status_code = 400

    def __init__(self, model: str) -> None:
        super().__init__("model_not_mapped")
        self.message = f"{model} is served by the remote upstream only (make remote-up)"


def selected_upstream(policy: Policy, kind: LlmUpstreamKind) -> AnyLlmUpstream:
    """The declared upstream ``kind`` selects. Raises `ValueError` when it is not declared, or
    when remote judges would have no model the remote upstream serves."""
    if kind is LlmUpstreamKind.LOCAL:
        return policy.upstreams.llm
    remote = policy.upstreams.llm_remote
    if remote is None:
        msg = "ACL_LLM_UPSTREAM=remote, but the policy declares no upstreams.llm_remote"
        raise ValueError(msg)
    if policy.judges is not None and remote.judge_model is None:
        msg = "ACL_LLM_UPSTREAM=remote with judges needs upstreams.llm_remote.judge_model"
        raise ValueError(msg)
    return remote


def require_upstream(kind: LlmUpstreamKind) -> Callable[[Policy], None]:
    """A `PolicyLoader` requirement: the selected upstream is declared (startup and reloads)."""

    def check(policy: Policy) -> None:
        selected_upstream(policy, kind)

    return check


def outgoing_request(payload: Mapping[str, Any], upstream: AnyLlmUpstream) -> dict[str, Any]:
    """The body sent to ``upstream`` for an approved request (always ``stream: false``)."""
    body = {key: value for key, value in payload.items() if key != "stream_options"}
    if isinstance(upstream, RemoteLlmUpstream):
        body = {key: value for key, value in body.items() if key in REMOTE_FORWARDED_FIELDS}
        logical = body.get("model")
        provider = upstream.provider_model(logical) if isinstance(logical, str) else None
        if provider is None:
            raise ModelNotMappedError(upstream, logical)
        body["model"] = provider
        body.update(cast("dict[str, Any]", copy.deepcopy(dict(upstream.extra_body))))
    if upstream.reasoning == "openrouter" and "reasoning_effort" in body:
        effort = body.pop("reasoning_effort")
        body["reasoning"] = {"enabled": False} if effort == "none" else {"effort": effort}
    body["stream"] = False
    return body


def incoming_completion(data: dict[str, Any], upstream: AnyLlmUpstream) -> dict[str, Any]:
    """The upstream's answer with its model named by logical id where the map says so.

    A provider id outside the map stays as reported, so ``model_allowlist`` sees a mismatch
    (fail closed) unless the operator lists it as an alias."""
    model = data.get("model")
    if isinstance(upstream, RemoteLlmUpstream) and isinstance(model, str):
        logical = upstream.logical_model(model)
        if logical is not None:
            return {**data, "model": logical}
    return data


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
    cost: object = None  # OpenRouter: USD charged; anything but a finite number >= 0 is ignored


def reported_cost_usd(value: object) -> float | None:
    """``usage.cost`` if it is a finite, non-negative number; else None (settle at the ceiling)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    cost = float(value)
    return cost if math.isfinite(cost) and cost >= 0 else None


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
        kind: LlmUpstreamKind = LlmUpstreamKind.LOCAL,
        env: Mapping[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._kind = kind
        self._env = env if env is not None else os.environ
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    @property
    def kind(self) -> LlmUpstreamKind:
        return self._kind

    def upstream(self, snapshot: PolicySnapshot) -> AnyLlmUpstream:
        """The selected upstream under ``snapshot`` (reloads cannot drop it: `require_upstream`)."""
        try:
            return selected_upstream(snapshot.policy, self._kind)
        except ValueError:
            logger.error("the selected LLM upstream (%s) is not declared", self._kind.value)  # noqa: TRY400 -- configuration message, no traceback needed
            raise UpstreamError("upstream_misconfigured") from None

    def host(self, snapshot: PolicySnapshot) -> str:
        return urlsplit(self.upstream(snapshot).base_url).hostname or ""

    def judge_model(self, snapshot: PolicySnapshot, configured: str) -> str:
        """The model a judge asks this upstream for: ``configured`` locally, the remote
        upstream's ``judge_model`` when remote is selected (local ids are not served there)."""
        upstream = self.upstream(snapshot)
        if isinstance(upstream, RemoteLlmUpstream) and upstream.judge_model is not None:
            return upstream.judge_model
        return configured

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
        upstream = self.upstream(snapshot)
        logical = cast("dict[str, Any]", payload).get("model")
        remote = snapshot.policy.upstreams.llm_remote
        remote_only = (
            remote is not None and isinstance(logical, str) and logical in remote.model_map
        )
        if remote_only and not isinstance(upstream, RemoteLlmUpstream):
            raise RemoteOnlyModelError(str(logical))
        body = outgoing_request(cast("dict[str, Any]", payload), upstream)
        started = time.perf_counter()
        sent = await self._send("POST", "/chat/completions", snapshot, body=body)
        elapsed = time.perf_counter() - started
        if isinstance(sent, dict) and "error" in sent:  # an error object with a 200 status
            logger.warning("upstream answered an error object with status 200")
            raise UpstreamError("upstream_error")
        try:
            completion = ChatCompletion.model_validate(sent)
        except ValidationError:
            raise UpstreamError("upstream_invalid_response") from None
        data = incoming_completion(cast("dict[str, Any]", sent), upstream)
        usage = (
            TokenUsage(
                model=str(logical) if isinstance(logical, str) else "",
                prompt_tokens=completion.usage.prompt_tokens,
                completion_tokens=completion.usage.completion_tokens,
                total_tokens=completion.usage.total_tokens,
                cost_usd=reported_cost_usd(completion.usage.cost),
            )
            if completion.usage is not None
            else None
        )
        return UpstreamResult(body=data, elapsed_s=elapsed, usage=usage)

    async def list_models(self, snapshot: PolicySnapshot) -> list[dict[str, Any]]:
        """The upstream's ``/models`` entries, as returned; for the remote upstream, its mapped
        logical ids only (a router lists hundreds of models the policy never named)."""
        upstream = self.upstream(snapshot)
        if isinstance(upstream, RemoteLlmUpstream):
            owner = urlsplit(upstream.base_url).hostname or "remote"
            return [
                {"id": logical, "object": "model", "owned_by": owner}
                for logical in upstream.model_map
            ]
        data = await self._send("GET", "/models", snapshot)
        try:
            models = ModelList.model_validate(data)
        except ValidationError:
            raise UpstreamError("upstream_invalid_response") from None
        return [card.model_dump(mode="json") for card in models.data]

    def _headers(self, upstream: AnyLlmUpstream) -> dict[str, str]:
        # identity only: a compressed body would be inflated before the size cap could see it.
        headers = {"accept": "application/json", "accept-encoding": "identity"}
        key_env = upstream.api_key_env
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
        upstream = self.upstream(snapshot)
        url = upstream.base_url.rstrip("/") + path
        raw = bytearray()
        try:
            async with self._client.stream(
                method,
                url,
                json=body,
                headers=self._headers(upstream),
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
            # NaN / Infinity are not JSON: read as null, so a stray one is ignored (a reported
            # cost settles at the ceiling) instead of crashing the response sent to the agent.
            return json.loads(raw, parse_constant=lambda _constant: None)
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
