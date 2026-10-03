"""The normalized envelope every call is turned into, and the verdicts controls return on it.

All models are frozen: a control receives a read-only view and the pipeline builds a new
`Interaction` (``model_copy(update=...)``) whenever a control rewrites the payload.
"""

import math
from datetime import datetime
from typing import Any, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from gateway.core.types import Action, Channel, Decision, SessionMode
from gateway.policy.permissions import Resource


class FrozenModel(BaseModel):
    """Base for boundary models: unknown fields rejected, instances immutable, numbers finite."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def cooldown_key(action: Action, resource: str) -> str:
    """Key under which a cooldown on one ``action:resource`` pair is stored."""
    return f"{action}:{resource}"


class Span(FrozenModel):
    """A region of a string inside the payload or result, addressed by JSON pointer.

    ``embedded`` addresses a string inside JSON text: the value at ``path`` is a JSON document
    serialized into a string (an OpenAI tool call's ``arguments``), ``embedded`` is a pointer
    into that document, and the offsets are in its decoded string. A span on a number (at
    ``path``, or ``embedded`` in one) replaces the whole number.
    """

    path: str
    start: int = Field(ge=0)
    end: int
    label: str = Field(min_length=1)
    embedded: str | None = None

    @field_validator("path", "embedded")
    @classmethod
    def _json_pointer(cls, value: str | None) -> str | None:
        if value and not value.startswith("/"):
            msg = f"span path must be a JSON pointer ('' or '/...'), got {value!r}"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _non_empty(self) -> Self:
        if self.end <= self.start:
            msg = f"span end ({self.end}) must be greater than start ({self.start})"
            raise ValueError(msg)
        return self


class FlaggedToolCall(FrozenModel):
    """A ``tool_call`` the intent judge flagged; the matching MCP call then needs approval."""

    tool: str = Field(min_length=1)
    args_digest: str = Field(min_length=1)


class Verdict(FrozenModel):
    """One control's opinion on one interaction at one stage."""

    decision: Decision
    control_id: str = Field(min_length=1)
    reason_code: str = Field(min_length=1)
    reason: str = ""  # human text, never contains payload data
    enforced: bool = True  # False = log_only: recorded and counted, decision not applied
    risk_delta: float = Field(default=0.0, ge=0.0, le=1.0)
    redactions: tuple[Span, ...] = ()
    rewrite: Any = None  # replacement payload (pre) or result (post)
    latency_ms: float = Field(default=0.0, ge=0.0)
    # Tool calls to flag in the session (``intent_judge``). A post ``require_approval`` that
    # names flags does not hold the released result: its approval obligation moves to the
    # matching MCP ``tools/call`` instead (SPEC "Intent vs enforcement").
    flags: tuple[FlaggedToolCall, ...] = ()
    # Taint the session even though the decision lets the call through: the control could not
    # show the content is clean and chose to allow it (``prompt_injection`` when the judge
    # cannot confirm a hit on the user's own prompt). A non-allow verdict of a tainting
    # control taints anyway; this is for allow verdicts. Recorded on the audit verdict.
    taint: bool = False


class Cooldown(FrozenModel):
    """A denied ``action:resource`` pair that stays denied until ``until``."""

    key: str = Field(min_length=1)
    until: AwareDatetime


class CallRecord(FrozenModel):
    """One past call of the session, kept for loop detection and the session trace."""

    at: AwareDatetime
    channel: Channel
    action: Action
    resource: str
    digest: str  # digest of the canonical arguments, never the arguments themselves


class SessionContext(FrozenModel):
    """Read-only snapshot of session state taken when the call was admitted.

    Mutating the session (risk, taint, timers) is the pipeline's job; controls and the
    evaluator only ever see this snapshot.
    """

    session_id: str = Field(min_length=1)
    principal: str = Field(min_length=1)
    actor: str = Field(min_length=1)
    mode: SessionMode
    risk: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_updated_at: AwareDatetime
    taint: bool = False
    created_at: AwareDatetime
    last_seen: AwareDatetime
    freeze_until: AwareDatetime | None = None
    cooldowns: tuple[Cooldown, ...] = ()
    flagged_tool_calls: tuple[FlaggedToolCall, ...] = ()
    # More flags than the session keeps were raised: the evicted ones are pending approval
    # obligations, so from then on every MCP tools/call needs approval (sticky).
    flags_overflowed: bool = False
    call_history: tuple[CallRecord, ...] = ()
    # The user's goal: the first user message the gateway forwarded to the LLM in this session,
    # set once and never from a later transcript (SPEC "Intent vs enforcement"). Raw text for
    # the intent judge only: never audited, logged or exported.
    goal: str | None = Field(default=None, repr=False)

    def risk_at(self, now: datetime, half_life_s: float) -> float:
        """Risk decayed exponentially from ``risk_updated_at`` to ``now``."""
        elapsed = max((now - self.risk_updated_at).total_seconds(), 0.0)
        return self.risk * math.pow(2.0, -elapsed / half_life_s)

    def cooldown_until(self, key: str, now: datetime) -> datetime | None:
        """End of the active cooldown on ``key``, or None when there is none."""
        active = [c.until for c in self.cooldowns if c.key == key and c.until > now]
        return max(active, default=None)

    def is_frozen(self, now: datetime) -> bool:
        return self.freeze_until is not None and self.freeze_until > now

    def is_flagged(self, tool: str, args_digest: str) -> bool:
        return FlaggedToolCall(tool=tool, args_digest=args_digest) in self.flagged_tool_calls


class Interaction(FrozenModel):
    """One normalized ``(action, resource)`` the pipeline authorizes and runs controls on."""

    session_id: str = Field(min_length=1)
    principal: str = Field(min_length=1)  # sub (human, or svc:<agent> for autonomous)
    actor: str = Field(min_length=1)  # act.sub
    mode: SessionMode
    channel: Channel
    action: Action
    resource: str  # "db:sales.orders", "http:api.stripe.com", "model:qwen3:8b"
    payload: Any  # request payload (pre) ...
    result: Any = None  # ... and upstream result (post)
    context: SessionContext
    server: str | None = None  # MCP upstream name (``/mcp/{server}``); None on other channels

    @field_validator("resource")
    @classmethod
    def _concrete_resource(cls, value: str) -> str:
        return str(Resource.parse(value))

    @property
    def cooldown_key(self) -> str:
        return cooldown_key(self.action, self.resource)


class RawCall(FrozenModel):
    """A request as it arrived at an entry point, before an adapter normalizes it."""

    channel: Channel
    data: dict[str, Any]  # parsed request body (OpenAI request, JSON-RPC message)
    server: str | None = None  # MCP upstream name from /mcp/{server}
    headers: dict[str, str] = Field(default_factory=dict[str, str])  # allowlisted subset only

    @model_validator(mode="after")
    def _server_iff_mcp(self) -> Self:
        if (self.channel is Channel.MCP) != (self.server is not None):
            msg = "server is required for MCP calls and forbidden otherwise"
            raise ValueError(msg)
        return self
