"""The upstream side of the pipeline: one `Upstream` per channel executes an approved call."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import Field

from gateway.core.envelope import FrozenModel
from gateway.errors import RejectionError
from gateway.policy.loader import PolicySnapshot


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


class UpstreamError(RejectionError):
    """The upstream failed. The agent gets a generic message: upstream text never leaks."""

    status_code = 502

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code, "the upstream service failed to answer")


class Upstream(ABC):
    """Executes the final (possibly rewritten) payload of an approved call exactly once."""

    @abstractmethod
    async def execute(self, payload: object, snapshot: PolicySnapshot) -> UpstreamResult: ...
