"""Shared vocabulary: the enums every layer of the gateway speaks."""

from enum import StrEnum


class Action(StrEnum):
    """What an interaction does to its resource."""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    EGRESS = "egress"
    GENERATE = "generate"


class Decision(StrEnum):
    """Outcome of a control or of the merged pipeline."""

    ALLOW = "allow"
    REDACT = "redact"
    BLOCK = "block"
    REQUIRE_APPROVAL = "require_approval"


class ControlMode(StrEnum):
    """How a control reacts to a detection. `log_only` is not a decision: it records only."""

    BLOCK = "block"
    REQUIRE_APPROVAL = "require_approval"
    REDACT = "redact"
    LOG_ONLY = "log_only"


# Most to least enforcing; profiles resolve a control's default mode along this order.
MODE_ENFORCEMENT_ORDER: tuple[ControlMode, ...] = (
    ControlMode.BLOCK,
    ControlMode.REQUIRE_APPROVAL,
    ControlMode.REDACT,
    ControlMode.LOG_ONLY,
)

# Strictest first; the merged decision is the first one any enforced verdict carries.
DECISION_STRICTNESS_ORDER: tuple[Decision, ...] = (
    Decision.BLOCK,
    Decision.REQUIRE_APPROVAL,
    Decision.REDACT,
    Decision.ALLOW,
)


class Stage(StrEnum):
    PRE = "pre"
    POST = "post"


class Channel(StrEnum):
    LLM = "llm"
    MCP = "mcp"
    A2A = "a2a"


class SessionMode(StrEnum):
    """Asserted by the token's `mode` claim and validated against the agent's `type`."""

    INTERACTIVE = "interactive"
    AUTONOMOUS = "autonomous"


class ControlKind(StrEnum):
    DETERMINISTIC = "deterministic"
    SEMANTIC = "semantic"


class Profile(StrEnum):
    """Strictness profile: sets the default mode of non-mandatory controls only."""

    STRICT = "strict"
    BALANCED = "balanced"
    PERMISSIVE = "permissive"


class LlmUpstreamKind(StrEnum):
    """Which declared LLM upstream serves ``generate`` and the judges (``ACL_LLM_UPSTREAM``).

    ``local`` (``upstreams.llm``, the product default) or ``remote`` (``upstreams.llm_remote``,
    an opt-in engineering upstream outside the machine)."""

    LOCAL = "local"
    REMOTE = "remote"
