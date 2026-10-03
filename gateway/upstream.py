"""The upstream side of the pipeline: one `Upstream` per channel executes an approved call."""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

import httpx
from pydantic import Field

from gateway.core.envelope import FrozenModel
from gateway.errors import RejectionError
from gateway.policy.loader import PolicySnapshot

logger = logging.getLogger(__name__)


class TokenUsage(FrozenModel):
    """Token counts the upstream reported for one call (``acl_tokens_total``)."""

    model: str
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class UpstreamResult(FrozenModel):
    body: Any = Field(repr=False)  # complete, buffered result; post controls see all of it
    elapsed_s: float = Field(ge=0.0)  # upstream wall time, excluded from gateway overhead
    usage: TokenUsage | None = None
    # The result came from an untrusted source (an MCP server with `trust: untrusted`): it
    # taints the session even when post controls block or replace it (SPEC "Risk score and taint").
    untrusted: bool = False


class UpstreamError(RejectionError):
    """The upstream failed. The agent gets a generic message: upstream text never leaks."""

    status_code = 502
    # True when content from an untrusted source may have reached the gateway before the
    # failure (an error message, a malformed or oversized body): it taints like a result.
    untrusted: bool = False

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code, "the upstream service failed to answer")


class Upstream(ABC):
    """Executes the final (possibly rewritten) payload of an approved call exactly once."""

    @abstractmethod
    async def execute(self, payload: object, snapshot: PolicySnapshot) -> UpstreamResult: ...


class DeadlineUpstream(Upstream):
    """Another upstream, bounded by a total wall-clock deadline (a call's GPU allowance).

    ``asyncio.timeout`` around the whole call, not an httpx timeout: httpx's timeouts bound
    each connect/read/write step, so a slow trickle could outlast them; this cannot.
    """

    def __init__(self, inner: Upstream, deadline_s: float) -> None:
        self._inner = inner
        self._deadline_s = deadline_s

    async def execute(self, payload: object, snapshot: PolicySnapshot) -> UpstreamResult:
        try:
            async with asyncio.timeout(self._deadline_s):
                return await self._inner.execute(payload, snapshot)
        except TimeoutError:
            raise UpstreamError("upstream_timeout") from None


def refuse_encoded(response: httpx.Response) -> None:
    """Refuse a content-coded body: inflating it would bypass ``max_response_bytes`` (a few
    KiB of gzip can expand to gigabytes). Upstream requests ask for ``identity``."""
    encoding = response.headers.get("content-encoding", "").strip().lower()
    if encoding not in {"", "identity"}:
        logger.warning("upstream answered with content-encoding %r", encoding[:32])
        raise UpstreamError("upstream_encoding_refused")


async def wire_chunks(response: httpx.Response) -> AsyncIterator[bytes]:
    """The body as received, never decompressed, for counting against a byte cap."""
    if response.is_stream_consumed:  # already buffered by the transport (e.g. in tests)
        yield response.content
        return
    async for chunk in response.aiter_raw():
        yield chunk
