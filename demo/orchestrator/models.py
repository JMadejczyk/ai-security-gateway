"""What the orchestrator reads and records: agent replies, audit entries, scene results."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


# ------------------------------------------------------------------ agent (acl_agent JSON lines


class ToolCall(Frozen):
    """One MCP ``tools/call`` the agent makes; ``approval_id`` retries a held call."""

    server: str
    tool: str
    arguments: dict[str, str]
    approval_id: str | None = None
    wait_throttle: bool = False  # the agent sleeps through Retry-After instead of giving up

    def retried_with(self, approval_id: str) -> "ToolCall":
        return self.model_copy(update={"approval_id": approval_id})

    def shown(self) -> str:
        """``server.tool(arg='value', ...)`` for the transcript."""
        args = ", ".join(f"{k}={v!r}" for k, v in self.arguments.items())
        return f"{self.server}.{self.tool}({args})"


class ToolReply(Frozen):
    """One ``python -m acl_agent mcp ...``: ``reason`` is ``ok`` or the gateway's reason code."""

    kind: Literal["mcp"]
    server: str
    tool: str
    status: int
    reason: str
    approval_id: str | None = None
    retry_after_s: float | None = None
    result: JsonValue = None
    throttle_waits: tuple[float, ...] = ()

    @property
    def ok(self) -> bool:
        return self.reason == "ok"


class ChatReply(Frozen):
    kind: Literal["chat"]
    status: int
    reason: str
    answer: str | None = None
    elapsed_s: float = 0.0


class ProbeReply(Frozen):
    kind: Literal["probe"]
    host: str
    port: int
    resolved: tuple[str, ...] = ()
    connected: bool
    error: str | None = None


type AgentReply = Annotated[ToolReply | ChatReply | ProbeReply, Field(discriminator="kind")]
_AGENT_REPLY: TypeAdapter[AgentReply] = TypeAdapter(AgentReply)


class AgentOutputError(ValueError):
    def __init__(self, output: str) -> None:
        super().__init__(f"the agent printed no JSON reply: {output[-300:]!r}")


def parse_agent_reply(stdout: str) -> AgentReply:
    """The agent's reply: the last non-empty line of its stdout, one JSON object."""
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise AgentOutputError(stdout)
    return _AGENT_REPLY.validate_json(lines[-1])


def first_count(result: JsonValue) -> int | None:
    """``count`` of the first row of a ``query`` result (``[{"count": 40}]``), if any."""
    if isinstance(result, list) and result and isinstance(result[0], dict):
        value = result[0].get("count")
        return value if isinstance(value, int) else None
    return None


# ------------------------------------------------------------------ operator side


class IssuedToken(Frozen):
    """``POST /auth/demo-token``'s answer (``session_id`` only for agent tokens)."""

    access_token: str = Field(repr=False)
    sub: str
    kind: str
    session_id: str | None = None
    agent: str | None = None
    mode: str | None = None


class ApprovalInfo(Frozen):
    """The fields of an ``/admin/approvals`` view the demo shows."""

    id: str
    state: str
    agent: str
    principal: str
    session_id: str
    server: str | None = None
    tool: str | None = None
    resources: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    decided_by: str | None = None
    outcome: str | None = None


class AuditVerdict(Frozen):
    control: str
    stage: str
    decision: str
    enforced: bool = True
    reason_code: str


class AuditLatency(Frozen):
    total: float | None = None
    upstream: float | None = None


class AuditEntry(Frozen):
    """One decision line of the gateway's audit JSONL (reason codes and metadata only)."""

    ts: str
    session_id: str
    principal: str
    actor: str
    channel: str
    action: str
    resource: str
    decision: str
    reason_code: str
    verdicts: tuple[AuditVerdict, ...] = ()
    effective_scope: tuple[str, ...] = ()
    risk: float = 0.0
    taint: bool = False
    policy_revision: str = ""
    feed_version: str | None = None
    latency_ms: AuditLatency = AuditLatency()
    approval_id: str | None = None

    def verdict(self, control: str, stage: str = "pre") -> AuditVerdict | None:
        """The verdict ``control`` gave at ``stage``, if it ran."""
        return next((v for v in self.verdicts if v.control == control and v.stage == stage), None)

    def deciding(self) -> tuple[AuditVerdict, ...]:
        """The verdicts that were not a plain allow: what made the decision."""
        return tuple(v for v in self.verdicts if v.decision != "allow")


class ReloadEvent(Frozen):
    """A ``policy_reload`` line of the audit log: the source of Grafana's annotations."""

    ts: str
    event: Literal["policy_reload"]
    result: str
    revision: str | None = None
    previous_revision: str | None = None


# ------------------------------------------------------------------ results


class Check(Frozen):
    """One expected outcome: ``actual`` must be one of ``expected``."""

    name: str
    expected: tuple[str, ...]
    actual: str

    @property
    def passed(self) -> bool:
        return self.actual in self.expected


class SceneResult(Frozen):
    number: int
    title: str
    checks: tuple[Check, ...] = ()
    sessions: dict[str, str] = Field(default_factory=dict[str, str])  # label -> session_id
    elapsed_s: float = 0.0
    error: str | None = None  # the scene could not run to the end

    @property
    def passed(self) -> bool:
        return self.error is None and bool(self.checks) and all(c.passed for c in self.checks)

    @property
    def failed_checks(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if not c.passed)


class DemoReport(Frozen):
    """Everything one run produced (``run_demo.py --json``)."""

    run_id: str
    scenes: tuple[SceneResult, ...]
    grafana_links: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return bool(self.scenes) and all(scene.passed for scene in self.scenes)
