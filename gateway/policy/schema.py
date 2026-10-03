"""Pydantic schema for `policy.yaml` (SPEC "Policy file" and "Schema rules").

Every model forbids unknown fields and is frozen. Cross-references (approver roles,
agent principals, tool mappings, control modes) are checked on the root `Policy`.
"""

import re
from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import (
    AfterValidator,
    Field,
    PositiveFloat,
    PositiveInt,
    StringConstraints,
    field_validator,
    model_validator,
)

from gateway.core.catalog import CONTROL_CATALOG, control_spec
from gateway.core.envelope import FrozenModel
from gateway.core.frozen import FrozenDict
from gateway.core.interfaces import ControlConfig
from gateway.core.types import Action, ControlMode, Profile, SessionMode
from gateway.policy.permissions import WILDCARD, PermissionSet, Resource

SERVICE_PRINCIPAL_PREFIX = "svc:"

# Role, agent and server names: no colon, so `svc:<agent>` and resource keys stay unambiguous.
type Name = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
type Threshold = Annotated[float, Field(ge=0.0, le=1.0)]
# Every timer and interval is positive and bounded, so `now + duration` can never overflow.
MAX_DURATION_S = 30 * 24 * 3600.0
type Duration = Annotated[float, Field(gt=0.0, le=MAX_DURATION_S)]
type EnvVarName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]

_PLACEHOLDER = re.compile(r"\{([^{}]*)\}")
_PLACEHOLDER_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _http_url(value: str) -> str:
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        msg = f"{value!r} must be an absolute http(s) URL"
        raise ValueError(msg)
    return value


type HttpUrl = Annotated[str, AfterValidator(_http_url)]


def _resource_template(value: str) -> str:
    """A concrete resource with ``{arg}`` placeholders, e.g. ``fs:reports/{name}``."""
    names = _PLACEHOLDER.findall(value)
    for name in names:
        if not _PLACEHOLDER_NAME.fullmatch(name):
            msg = f"resource template {value!r}: placeholder {{{name}}} is not an argument name"
            raise ValueError(msg)
    skeleton = _PLACEHOLDER.sub("x", value)
    if "{" in skeleton or "}" in skeleton:
        msg = f"resource template {value!r} has an unbalanced brace"
        raise ValueError(msg)
    namespace = value.partition(":")[0]
    if "{" in namespace:
        msg = f"resource template {value!r}: the namespace must be literal"
        raise ValueError(msg)
    Resource.parse(skeleton)  # same shape as a concrete resource, no wildcard
    return value


type ResourceTemplate = Annotated[str, AfterValidator(_resource_template)]


# --------------------------------------------------------------------------- upstreams


class LlmUpstream(FrozenModel):
    base_url: HttpUrl
    api_key_env: EnvVarName | None = None


class McpTool(FrozenModel):
    """Operator-owned mapping of one MCP tool to ``(action, resource)``."""

    action: Action
    resource: ResourceTemplate | None = None


type Adapter = Literal["sql", "http", "fs", "generic"]
type Trust = Literal["internal", "untrusted"]


class McpServer(FrozenModel):
    url: HttpUrl
    adapter: Adapter
    trust: Trust
    tools: FrozenDict[Name, McpTool] = Field(default_factory=FrozenDict[str, McpTool])
    # `tool_pinning`: without `pins/<server>.json` every tool of the server is blocked
    # (`tool_not_pinned`) unless the operator opts the server out explicitly with false.
    require_pin: bool = True

    @model_validator(mode="after")
    def _resource_per_adapter(self) -> Self:
        for name, tool in self.tools.items():
            if self.adapter == "sql" and tool.resource is not None:
                msg = (
                    f"tool {name!r}: the sql adapter derives resources from the parsed tables, "
                    "a resource template would never be checked"
                )
                raise ValueError(msg)
            if self.adapter != "sql" and tool.resource is None:
                msg = f"tool {name!r}: the {self.adapter} adapter needs a resource template"
                raise ValueError(msg)
        return self


class Upstreams(FrozenModel):
    llm: LlmUpstream
    mcp: FrozenDict[Name, McpServer] = Field(default_factory=FrozenDict[str, McpServer])


# --------------------------------------------------------------------- roles and agents


class Role(FrozenModel):
    allow: PermissionSet = PermissionSet()


class Agent(FrozenModel):
    type: SessionMode
    max_actions: tuple[Action, ...]
    allow: PermissionSet = PermissionSet()
    deny: PermissionSet = PermissionSet()
    approvers: tuple[Name, ...] = ()
    principals: tuple[str, ...] | None = None  # None = default rule for the agent type

    @field_validator("max_actions", "approvers")
    @classmethod
    def _unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            msg = f"duplicate entries in {list(value)}"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _allow_within_max_actions(self) -> Self:
        # A concrete action outside max_actions could never be used: reject the dead grant
        # instead of silently filtering it. `*` is fine, it is narrowed at match time.
        for permission in self.allow:
            if permission.action != WILDCARD and permission.action not in self.max_actions:
                msg = (
                    f"allow entry {str(permission)!r} uses action {str(permission.action)!r} "
                    f"outside max_actions {[str(a) for a in self.max_actions]}"
                )
                raise ValueError(msg)
        return self

    @property
    def grants(self) -> PermissionSet:
        """The agent's allow list filtered by ``max_actions`` (the A in the decision model)."""
        return self.allow.restricted_to(self.max_actions)


# --------------------------------------------------------------------------- risk rules


class Throttle(FrozenModel):
    max_actions: PositiveInt
    per_s: Duration


class When(FrozenModel):
    """Condition of a risk rule; every present field must hold."""

    taint: bool | None = None
    risk_gt: Threshold | None = None  # strict: > threshold

    @model_validator(mode="after")
    def _at_least_one(self) -> Self:
        if self.taint is None and self.risk_gt is None:
            msg = "a risk rule `when` needs at least one condition (taint, risk_gt)"
            raise ValueError(msg)
        return self

    def holds(self, *, risk: float, taint: bool) -> bool:
        if self.taint is not None and self.taint != taint:
            return False
        return self.risk_gt is None or risk > self.risk_gt


class Then(FrozenModel):
    """Effects of a risk rule. They only remove or condition, never grant."""

    deny_actions: tuple[Action, ...] = ()
    actions: tuple[Action, ...] = ()
    mode: Literal["require_approval"] | None = None
    cooldown_s: Duration | None = None
    freeze_tools: bool = False
    duration_s: Duration | None = None
    throttle: Throttle | None = None
    alert: bool = False

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if bool(self.actions) != (self.mode is not None):
            msg = "`actions` and `mode` must be given together"
            raise ValueError(msg)
        if self.freeze_tools != (self.duration_s is not None):
            msg = "`freeze_tools: true` needs `duration_s` (and `duration_s` needs freeze_tools)"
            raise ValueError(msg)
        effects = (
            self.deny_actions,
            self.actions,
            self.cooldown_s,
            self.freeze_tools,
            self.throttle,
            self.alert,
        )
        if not any(effect for effect in effects):
            msg = "a risk rule `then` needs at least one effect"
            raise ValueError(msg)
        return self


class RiskRule(FrozenModel):
    when: When
    then: Then


class RiskRules(FrozenModel):
    """Profile-independent session restrictions, one ordered list per session mode."""

    interactive: tuple[RiskRule, ...]
    autonomous: tuple[RiskRule, ...]

    def for_mode(self, mode: SessionMode) -> tuple[RiskRule, ...]:
        match mode:
            case SessionMode.INTERACTIVE:
                return self.interactive
            case SessionMode.AUTONOMOUS:
                return self.autonomous


class Risk(FrozenModel):
    half_life_s: Duration = 600.0


# ------------------------------------------------------------------ operational sections


class Approvals(FrozenModel):
    timeout_s: Duration = 600.0
    on_timeout: Literal["deny"] = "deny"


class Sessions(FrozenModel):
    idle_ttl_s: Duration = 3600.0
    max_lifetime_s: Duration = 86400.0

    @model_validator(mode="after")
    def _idle_within_lifetime(self) -> Self:
        if self.idle_ttl_s > self.max_lifetime_s:
            msg = "sessions.idle_ttl_s cannot exceed sessions.max_lifetime_s"
            raise ValueError(msg)
        return self


class ThrottleBackoff(FrozenModel):
    backoff: Literal["exponential"] = "exponential"
    base_s: Duration = 5.0
    max_s: Duration = 300.0

    @model_validator(mode="after")
    def _base_within_max(self) -> Self:
        if self.base_s > self.max_s:
            msg = "throttle.base_s cannot exceed throttle.max_s"
            raise ValueError(msg)
        return self


class Blocklist(FrozenModel):
    users: tuple[str, ...] = ()
    agents: tuple[str, ...] = ()
    use_cases: PermissionSet = PermissionSet()  # permission patterns


class Limits(FrozenModel):
    max_request_bytes: PositiveInt = 1_048_576
    max_response_bytes: PositiveInt = 4_194_304
    upstream_timeout_s: Duration = 120.0
    # Completion cap the gateway puts on an LLM request that names none, so the budget
    # reservation (prompt estimate + completion cap) is an upper bound on what the call uses.
    default_max_tokens: PositiveInt = 4096
    # The largest completion cap any LLM request may hold; larger requested caps are lowered.
    max_completion_tokens: PositiveInt = 32_768

    @model_validator(mode="after")
    def _default_within_max(self) -> Self:
        if self.default_max_tokens > self.max_completion_tokens:
            msg = "limits.default_max_tokens cannot exceed limits.max_completion_tokens"
            raise ValueError(msg)
        return self


# ------------------------------------------------------------------------------ budgets


class DailyBudget(FrozenModel):
    daily_tokens: PositiveInt | None = None
    daily_cost_usd: PositiveFloat | None = None
    daily_tool_calls: PositiveInt | None = None
    daily_gpu_seconds: PositiveFloat | None = None


class SessionBudget(FrozenModel):
    tokens: PositiveInt | None = None
    cost_usd: PositiveFloat | None = None
    tool_calls: PositiveInt | None = None
    gpu_seconds: PositiveFloat | None = None


class Budgets(FrozenModel):
    per_user: DailyBudget = DailyBudget()
    per_agent: DailyBudget = DailyBudget()
    per_session: SessionBudget = SessionBudget()
    soft_limit_pct: Annotated[float, Field(gt=0.0, le=100.0)] = 80.0


type Price = Annotated[float, Field(ge=0.0)]  # USD; finite (FrozenModel forbids inf/nan)


class ModelPrice(FrozenModel):
    """What one model costs. ``gpu_second`` prices upstream wall time, an estimate of the GPU
    time a local model (Ollama) spends on a call."""

    prompt_per_1k: Price = 0.0
    completion_per_1k: Price = 0.0
    gpu_second: Price = 0.0


def _model_identifier(value: str) -> str:
    Resource.parse(f"model:{value}")  # the identifier a `model:<id>` resource carries
    return value


# Keyed by the model identifier the call is authorized for (``generate:model:<id>``). A model
# without an entry costs nothing, but its tokens and GPU time still count against budgets.
type PricedModel = Annotated[str, AfterValidator(_model_identifier)]


# ----------------------------------------------------------------------------- controls


class PiiEntity(StrEnum):
    """Entity IDs the ``pii`` control detects (Presidio's names, plus our Polish ones)."""

    PL_PESEL = "PL_PESEL"
    PL_NIP = "PL_NIP"
    IBAN_CODE = "IBAN_CODE"
    EMAIL_ADDRESS = "EMAIL_ADDRESS"
    PHONE_NUMBER = "PHONE_NUMBER"


class PiiConfig(ControlConfig):
    threshold: Threshold = 0.6
    entities: tuple[PiiEntity, ...] = tuple(PiiEntity)  # every entity when omitted

    @field_validator("entities")
    @classmethod
    def _entities_unique(cls, value: tuple[PiiEntity, ...]) -> tuple[PiiEntity, ...]:
        if len(set(value)) != len(value):
            msg = f"duplicate entries in {[str(e) for e in value]}"
            raise ValueError(msg)
        return value


class PromptInjectionConfig(ControlConfig):
    threshold: Threshold = 0.85
    judge_band: tuple[Threshold, Threshold] = (0.5, 0.85)
    # Characters of not yet classified text one call may carry; more fails closed. The model
    # runs at roughly 5 KB/s on 4 CPU threads, so this also bounds the added latency.
    max_chars: Annotated[int, Field(gt=0, le=10_000_000)] = 50_000

    @field_validator("judge_band")
    @classmethod
    def _ordered(cls, value: tuple[float, float]) -> tuple[float, float]:
        low, high = value
        if low > high:
            msg = f"judge_band {list(value)} must be ordered [low, high]"
            raise ValueError(msg)
        return value


class ToolPoisoningConfig(ControlConfig):
    threshold: Threshold = 0.85  # classifier score at which a tool definition is poisoned


class SqlGuardConfig(ControlConfig):
    max_cost: PositiveFloat = 10_000.0
    force_limit: PositiveInt = 500
    timeout_ms: Annotated[int, Field(gt=0, le=int(MAX_DURATION_S * 1000))] = 3_000
    max_result_bytes: PositiveInt = 1_048_576  # the SQL server's cap on one serialized result


class SignaturesConfig(ControlConfig):
    feed: str | None = Field(default=None, min_length=1)  # URL or file path
    refresh_s: Duration = 30.0


class LoopDetectConfig(ControlConfig):
    max_repeats: PositiveInt = 5
    window_s: Duration = 60.0


type ModelName = Annotated[str, StringConstraints(min_length=1, max_length=256)]


class ModelAllowlistConfig(ControlConfig):
    """Other names the upstream may report for a requested model (a router alias resolving to
    a provider model). A requested name without a tag also accepts its ``:latest`` (Ollama)."""

    aliases: FrozenDict[ModelName, tuple[ModelName, ...]] = Field(
        default_factory=FrozenDict[str, tuple[str, ...]]
    )


class IntentJudgeConfig(ControlConfig):
    model: str | None = Field(default=None, min_length=1)  # overrides `judges.model`


class Judges(FrozenModel):
    """The one `JudgeClient` every LLM judge shares (SPEC "Judges").

    Judges call ``upstreams.llm`` directly with ``model``: they are not agent calls, so the
    model needs no grant and no ``pricing`` entry, and judge tokens are never charged to an
    agent's budget. Without this section the judge-backed controls are off (``intent_judge``
    and ``output_policy`` allow with ``judge_not_configured``), and configuring any of them
    explicitly (or ``prompt_injection.judge_band``) is a validation error. With it, a judge
    that cannot answer in time fails its control closed.
    """

    model: ModelName
    timeout_s: Annotated[float, Field(gt=0.0, le=300.0)] = 10.0  # total deadline per call
    # Content longer than this is not judged at all: the judge is unavailable for it (fail
    # closed), never handed a truncated view that could hide the part that matters.
    max_content_chars: Annotated[int, Field(gt=0, le=1_000_000)] = 16_000
    max_output_tokens: Annotated[int, Field(gt=0, le=32_768)] = 1024


class Controls(FrozenModel):
    """Per-control settings. One field per catalog ID, so unknown IDs are rejected."""

    authn: ControlConfig | None = None
    authz: ControlConfig | None = None
    model_allowlist: ModelAllowlistConfig | None = None
    pii: PiiConfig | None = None
    secrets: ControlConfig | None = None
    sql_guard: SqlGuardConfig | None = None
    egress: ControlConfig | None = None
    signatures: SignaturesConfig | None = None
    tool_pinning: ControlConfig | None = None
    budget: ControlConfig | None = None
    loop_detect: LoopDetectConfig | None = None
    prompt_injection: PromptInjectionConfig | None = None
    tool_poisoning: ToolPoisoningConfig | None = None
    intent_judge: IntentJudgeConfig | None = None
    output_policy: ControlConfig | None = None

    def configured(self, control_id: str) -> ControlConfig | None:
        control_spec(control_id)  # unknown IDs raise
        config: ControlConfig | None = getattr(self, control_id)
        return config


# Typed settings per control; every other control takes the plain ControlConfig.
_CONFIG_TYPES: Mapping[str, type[ControlConfig]] = MappingProxyType(
    {
        "model_allowlist": ModelAllowlistConfig,
        "pii": PiiConfig,
        "sql_guard": SqlGuardConfig,
        "signatures": SignaturesConfig,
        "loop_detect": LoopDetectConfig,
        "prompt_injection": PromptInjectionConfig,
        "tool_poisoning": ToolPoisoningConfig,
        "intent_judge": IntentJudgeConfig,
    }
)

# Keep the schema and the catalog from drifting apart.
if set(Controls.model_fields) != set(CONTROL_CATALOG):  # pragma: no cover - import-time guard
    _drift = set(Controls.model_fields) ^ set(CONTROL_CATALOG)
    _msg = f"Controls schema and CONTROL_CATALOG disagree on {sorted(_drift)}"
    raise RuntimeError(_msg)


# --------------------------------------------------------------------------------- root


class Policy(FrozenModel):
    schema_version: Literal[2]
    profile: Profile
    default: Literal["deny"]
    upstreams: Upstreams
    roles: FrozenDict[Name, Role] = Field(default_factory=FrozenDict[str, Role])
    agents: FrozenDict[Name, Agent] = Field(default_factory=FrozenDict[str, Agent])
    risk: Risk = Risk()
    risk_rules: RiskRules
    approvals: Approvals = Approvals()
    sessions: Sessions = Sessions()
    throttle: ThrottleBackoff = ThrottleBackoff()
    blocklist: Blocklist = Blocklist()
    limits: Limits = Limits()
    controls: Controls = Controls()
    budgets: Budgets = Budgets()
    pricing: FrozenDict[PricedModel, ModelPrice] = Field(
        default_factory=FrozenDict[str, ModelPrice]
    )
    judges: Judges | None = None

    @model_validator(mode="after")
    def _approver_roles_exist(self) -> Self:
        for agent_id, agent in self.agents.items():
            if missing := [role for role in agent.approvers if role not in self.roles]:
                msg = f"agent {agent_id!r}: approver roles {missing} are not defined in roles"
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _principals_match_agent_type(self) -> Self:
        for agent_id, agent in self.agents.items():
            principals = agent.principals or ()
            if agent.type is SessionMode.AUTONOMOUS:
                expected = service_principal(agent_id)
                if any(p != expected for p in principals):
                    msg = f"autonomous agent {agent_id!r} can only act as {expected!r}"
                    raise ValueError(msg)
            elif any(p.startswith(SERVICE_PRINCIPAL_PREFIX) for p in principals):
                msg = f"interactive agent {agent_id!r} cannot list service principals"
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _judge_controls_have_a_judge(self) -> Self:
        """A judge-backed control configured explicitly needs ``judges:``: the operator asked
        for it, and silently running it off (or failing closed on every call) would be a
        misconfiguration either way. Left out of ``controls`` with no ``judges:``, the judge
        controls are off (``judge_not_configured``)."""
        if self.judges is not None:
            return self
        controls = self.controls
        configured = [
            name
            for name, config in (
                ("intent_judge", controls.intent_judge),
                ("output_policy", controls.output_policy),
            )
            if config is not None
        ]
        injection = controls.prompt_injection
        if injection is not None and "judge_band" in injection.model_fields_set:
            configured.append("prompt_injection.judge_band")
        if configured:
            msg = f"{configured} need an LLM judge: add a `judges:` section (judges.model)"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _control_modes_supported(self) -> Self:
        for control_id, spec in CONTROL_CATALOG.items():
            config = self.controls.configured(control_id)
            if config is None or config.mode is None:
                continue
            if spec.mandatory and config.mode is ControlMode.LOG_ONLY:
                msg = f"control {control_id!r} is mandatory and cannot be set to log_only"
                raise ValueError(msg)
            if not spec.supports(config.mode):
                supported = [str(m) for m in spec.modes]
                msg = (
                    f"control {control_id!r} does not support mode {str(config.mode)!r} "
                    f"(supported: {supported})"
                )
                raise ValueError(msg)
        return self

    def control_config(self, control_id: str) -> ControlConfig:
        """Configured settings, or the control's defaults when it is omitted."""
        configured = self.controls.configured(control_id)
        if configured is not None:
            return configured
        return _CONFIG_TYPES.get(control_id, ControlConfig)()

    def resolved_control_mode(self, control_id: str) -> ControlMode:
        """Effective mode of a control under this policy's profile.

        Omitted controls are active with their profile default; mandatory controls are
        always active and never log_only.
        """
        configured = self.controls.configured(control_id)
        return control_spec(control_id).resolve_mode(
            self.profile, configured.mode if configured is not None else None
        )

    def control_risk_delta(self, control_id: str) -> float:
        configured = self.controls.configured(control_id)
        if configured is not None and configured.risk_delta is not None:
            return configured.risk_delta
        return control_spec(control_id).default_risk_delta


def service_principal(agent_id: str) -> str:
    """The principal an autonomous agent acts as."""
    return f"{SERVICE_PRINCIPAL_PREFIX}{agent_id}"
