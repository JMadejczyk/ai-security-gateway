"""Control catalog: metadata for every control ID in SPEC "Control catalog".

The policy schema validates `controls:` against it and profile resolution uses each
control's supported modes, ordered from most to least enforcing. The control classes
themselves live in `gateway/controls/`; this module is metadata only.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Self

from pydantic import Field, model_validator

from gateway.core.envelope import FrozenModel
from gateway.core.types import (
    MODE_ENFORCEMENT_ORDER,
    Channel,
    ControlKind,
    ControlMode,
    Profile,
    Stage,
)

_ALL_CHANNELS = frozenset(Channel)


class ControlSpec(FrozenModel):
    """What the policy needs to know about one control."""

    id: str = Field(min_length=1)
    kind: ControlKind
    stages: frozenset[Stage] = Field(min_length=1)
    channels: frozenset[Channel] = Field(min_length=1)
    modes: tuple[ControlMode, ...] = Field(min_length=1)  # most -> least enforcing
    mandatory: bool = False
    default_risk_delta: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _modes_consistent(self) -> Self:
        if len(set(self.modes)) != len(self.modes):
            msg = f"control {self.id!r} lists a mode twice"
            raise ValueError(msg)
        ranked = sorted(self.modes, key=MODE_ENFORCEMENT_ORDER.index)
        if list(self.modes) != ranked:
            msg = f"control {self.id!r} modes must be ordered most to least enforcing"
            raise ValueError(msg)
        if self.mandatory and ControlMode.LOG_ONLY in self.modes:
            msg = f"mandatory control {self.id!r} cannot support log_only"
            raise ValueError(msg)
        return self

    @property
    def most_enforcing(self) -> ControlMode:
        return self.modes[0]

    @property
    def least_enforcing(self) -> ControlMode:
        return self.modes[-1]

    def supports(self, mode: ControlMode) -> bool:
        return mode in self.modes

    def resolve_mode(self, profile: Profile, configured: ControlMode | None) -> ControlMode:
        """Effective mode per SPEC "Strictness profiles".

        An explicitly configured mode always wins. Mandatory controls ignore the profile and
        fall back to their most enforcing mode.
        """
        if configured is not None:
            return configured
        if self.mandatory:
            return self.most_enforcing
        match profile:
            case Profile.STRICT:
                return self.most_enforcing
            case Profile.BALANCED:
                for preferred in (ControlMode.REDACT, ControlMode.REQUIRE_APPROVAL):
                    if self.supports(preferred):
                        return preferred
                return self.most_enforcing
            case Profile.PERMISSIVE:
                return self.least_enforcing


def _spec(  # noqa: PLR0913 -- one keyword per catalog column keeps the table below readable
    control_id: str,
    kind: ControlKind,
    stages: set[Stage],
    modes: tuple[ControlMode, ...],
    *,
    channels: frozenset[Channel] = _ALL_CHANNELS,
    mandatory: bool = False,
    risk_delta: float = 0.0,
) -> ControlSpec:
    return ControlSpec(
        id=control_id,
        kind=kind,
        stages=frozenset(stages),
        channels=channels,
        modes=modes,
        mandatory=mandatory,
        default_risk_delta=risk_delta,
    )


_DET, _SEM = ControlKind.DETERMINISTIC, ControlKind.SEMANTIC
_PRE, _POST = Stage.PRE, Stage.POST
_BLOCK, _APPROVAL = ControlMode.BLOCK, ControlMode.REQUIRE_APPROVAL
_REDACT, _LOG = ControlMode.REDACT, ControlMode.LOG_ONLY
_LLM, _MCP = frozenset({Channel.LLM}), frozenset({Channel.MCP})

CONTROL_CATALOG: Mapping[str, ControlSpec] = MappingProxyType(
    {
        spec.id: spec
        for spec in (
            _spec("authn", _DET, {_PRE}, (_BLOCK,), mandatory=True),
            _spec("authz", _DET, {_PRE}, (_BLOCK, _APPROVAL), mandatory=True, risk_delta=0.1),
            _spec("model_allowlist", _DET, {_PRE, _POST}, (_BLOCK, _LOG), channels=_LLM),
            _spec("pii", _DET, {_PRE, _POST}, (_BLOCK, _REDACT, _LOG), risk_delta=0.1),
            _spec(
                "secrets", _DET, {_PRE, _POST}, (_BLOCK, _REDACT), mandatory=True, risk_delta=0.3
            ),
            _spec("sql_guard", _DET, {_PRE}, (_BLOCK,), channels=_MCP, mandatory=True),
            # An attempt to reach an internal address is a threat signal, like a leaked secret.
            _spec("egress", _DET, {_PRE}, (_BLOCK, _APPROVAL), channels=_MCP, risk_delta=0.3),
            _spec("signatures", _DET, {_PRE, _POST}, (_BLOCK, _LOG), risk_delta=0.4),
            _spec("tool_pinning", _DET, {_PRE}, (_BLOCK,), channels=_MCP),
            _spec("budget", _DET, {_PRE, _POST}, (_BLOCK,)),
            _spec("loop_detect", _DET, {_PRE}, (_BLOCK,)),
            _spec(
                "prompt_injection",
                _SEM,
                {_PRE, _POST},
                (_BLOCK, _LOG),
                channels=_LLM | _MCP,
                risk_delta=0.6,
            ),
            _spec("tool_poisoning", _SEM, {_PRE}, (_BLOCK,), channels=_MCP),
            _spec("intent_judge", _SEM, {_POST}, (_APPROVAL, _LOG), channels=_LLM),
            _spec("output_policy", _SEM, {_POST}, (_BLOCK, _REDACT), channels=_LLM),
        )
    }
)

MANDATORY_CONTROLS: frozenset[str] = frozenset(
    spec.id for spec in CONTROL_CATALOG.values() if spec.mandatory
)


def control_spec(control_id: str) -> ControlSpec:
    """Catalog entry for ``control_id``; unknown IDs are a programming or policy error."""
    try:
        return CONTROL_CATALOG[control_id]
    except KeyError:
        msg = f"unknown control id {control_id!r}"
        raise KeyError(msg) from None
