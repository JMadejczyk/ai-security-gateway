"""The agent's two gateway channels: MCP (streamable HTTP) and OpenAI-compatible chat.

Standard library + httpx only, so the same code runs in the hardened agent container and on
the host (``observability.smoke_traffic``). The gateway answers a refused MCP call with a tool
error whose ``_meta`` carries ``ai-control-layer/reason_code`` (and ``approval_id`` when the
call is held for approval); a refused chat call is an OpenAI-style error with that code.
"""

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final, Self, cast

import httpx

PROTOCOL_VERSION: Final = "2025-06-18"
META: Final = "ai-control-layer/"
APPROVAL_META: Final = f"{META}approval_id"
REASON_META: Final = f"{META}reason_code"
THROTTLED: Final = "throttled"
MAX_THROTTLE_RETRIES: Final = 6
DEFAULT_RETRY_AFTER_S: Final = 10.0
HTTP_OK: Final = 200

type JsonObject = dict[str, object]


def as_object(value: object) -> JsonObject:
    """``value`` if it is a JSON object, else an empty one."""
    return cast(JsonObject, value) if isinstance(value, dict) else {}


@dataclass(frozen=True)
class Outcome:
    """What one call got: ``reason`` is ``ok`` or the gateway's reason code."""

    status: int
    reason: str
    approval_id: str | None = None
    retry_after_s: float | None = None
    result: object = None  # the tool's structured result (or text) when the call succeeded
    throttle_waits: tuple[float, ...] = field(default=())  # seconds slept on Retry-After

    @property
    def ok(self) -> bool:
        return self.reason == "ok"

    @property
    def throttled(self) -> bool:
        return self.reason == THROTTLED


def parse_rpc(response: httpx.Response) -> JsonObject:
    """The JSON-RPC message in a plain JSON or a one-event SSE answer."""
    if not response.content:
        return {}
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        for line in response.text.splitlines():
            if line.startswith("data:") and line[5:].strip():
                return as_object(httpx.Response(HTTP_OK, content=line[5:]).json())
        return {}
    return as_object(response.json())


def tool_outcome(status: int, body: JsonObject, retry_after_s: float | None = None) -> Outcome:
    """A ``tools/call`` answer as an `Outcome`."""
    if "error" in body:  # refused before any tool ran: the reason code is in error.data
        error = as_object(body["error"])
        reason = as_object(error.get("data")).get("reason_code") or error.get("message", "error")
        return Outcome(status=status, reason=str(reason), retry_after_s=retry_after_s)
    result = as_object(body.get("result"))
    if not result.get("isError"):
        structured = as_object(result.get("structuredContent"))
        value: object = structured.get("result", structured or None)
        if value is None:
            texts = [as_object(c).get("text") for c in cast(list[object], result.get("content"))]
            value = "\n".join(str(t) for t in texts if t is not None)
        return Outcome(status=status, reason="ok", result=value)
    meta = as_object(result.get("_meta"))
    approval = meta.get(APPROVAL_META)
    return Outcome(
        status=status,
        reason=str(meta.get(REASON_META, "tool_error")),
        approval_id=str(approval) if approval is not None else None,
        retry_after_s=retry_after_s,
    )


class MCPSession:
    """One downstream MCP session on ``/mcp/<server>`` of the gateway's agent listener."""

    def __init__(
        self, client: httpx.Client, token: str, server: str, *, name: str = "databot"
    ) -> None:
        self._client = client
        self._path = f"/mcp/{server}"
        self._name = name
        self._headers = {
            "authorization": f"Bearer {token}",
            "accept": "application/json, text/event-stream",
            "content-type": "application/json",
        }
        self._next_id = 0

    def __enter__(self) -> Self:
        status, body, _ = self._rpc(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": self._name, "version": "1"},
            },
        )
        if "result" not in body:
            msg = f"MCP initialize on {self._path} failed ({status}): {body}"
            raise RuntimeError(msg)
        self._headers["mcp-protocol-version"] = PROTOCOL_VERSION
        self._client.post(
            self._path,
            headers=self._headers,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        ).raise_for_status()
        return self

    def __exit__(self, *_: object) -> None:
        with contextlib.suppress(httpx.HTTPError):
            self._client.delete(self._path, headers=self._headers)

    def call(self, tool: str, approval_id: str | None = None, **arguments: object) -> Outcome:
        """``tools/call``; ``approval_id`` retries a held call (``_meta`` approval protocol)."""
        params: JsonObject = {"name": tool, "arguments": arguments}
        if approval_id is not None:
            params["_meta"] = {APPROVAL_META: approval_id}
        status, body, retry_after = self._rpc("tools/call", params)
        return tool_outcome(status, body, retry_after)

    def _rpc(self, method: str, params: JsonObject) -> tuple[int, JsonObject, float | None]:
        self._next_id += 1
        response = self._client.post(
            self._path,
            headers=self._headers,
            json={"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params},
        )
        if session := response.headers.get("mcp-session-id"):
            self._headers["mcp-session-id"] = session
        retry = response.headers.get("retry-after")
        return response.status_code, parse_rpc(response), float(retry) if retry else None


@dataclass(frozen=True)
class ChatOutcome:
    """One chat completion: the answer text, or the gateway's reason code."""

    status: int
    reason: str
    answer: str | None = None
    elapsed_s: float = 0.0


def chat(  # noqa: PLR0913 -- the OpenAI request knobs the demo sets, all keyword-only
    client: httpx.Client,
    token: str,
    text: str,
    *,
    model: str = "qwen3:8b",
    max_tokens: int = 80,
    reasoning_effort: str | None = "none",
) -> ChatOutcome:
    """One user turn on ``/v1/chat/completions``. ``reasoning_effort: none`` keeps qwen3 from
    thinking for minutes on CPU; ``max_tokens`` bounds the answer (and the budget reserved)."""
    body: JsonObject = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [{"role": "user", "content": text}],
    }
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
    started = time.monotonic()
    response = client.post(
        "/v1/chat/completions", headers={"authorization": f"Bearer {token}"}, json=body
    )
    elapsed = round(time.monotonic() - started, 1)
    payload = as_object(response.json()) if response.content else {}
    if response.status_code == HTTP_OK:
        choices = cast(list[object], payload.get("choices") or [{}])
        message = as_object(as_object(choices[0]).get("message"))
        content = message.get("content")
        return ChatOutcome(
            status=HTTP_OK,
            reason="ok",
            answer=str(content) if content is not None else None,
            elapsed_s=elapsed,
        )
    error = as_object(payload.get("error"))
    return ChatOutcome(
        status=response.status_code, reason=str(error.get("code", "error")), elapsed_s=elapsed
    )


def unthrottled(
    call: Callable[[], Outcome],
    *,
    sleep: Callable[[float], None] = time.sleep,
    retries: int = MAX_THROTTLE_RETRIES,
) -> Outcome:
    """Retry a call the autonomous throttle rejected, after each Retry-After (the agent slows
    down instead of stopping). The returned outcome lists the waits."""
    outcome = call()
    waits: list[float] = []
    for _ in range(retries):
        if not outcome.throttled:
            break
        wait = (outcome.retry_after_s or DEFAULT_RETRY_AFTER_S) + 0.5
        waits.append(wait)
        sleep(wait)
        outcome = call()
    if not waits:
        return outcome
    return Outcome(
        status=outcome.status,
        reason=outcome.reason,
        approval_id=outcome.approval_id,
        retry_after_s=outcome.retry_after_s,
        result=outcome.result,
        throttle_waits=tuple(waits),
    )
