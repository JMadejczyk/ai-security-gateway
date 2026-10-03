"""The two extension points of the core (SPEC "Pipeline and interfaces").

A new tool means a new `Adapter`; a new guardrail means a new `Control` plus a YAML entry.
"""

from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import Field

from gateway.core.envelope import FrozenModel, Interaction, RawCall, SessionContext, Verdict
from gateway.core.types import ControlKind, ControlMode, Stage


class ControlConfig(FrozenModel):
    """Settings every control accepts; controls with more settings subclass this.

    ``mode`` left unset means "resolve from the profile"; ``risk_delta`` left unset means
    the catalog default.
    """

    mode: ControlMode | None = None
    risk_delta: float | None = Field(default=None, ge=0.0, le=1.0)


class Adapter(ABC):
    """Turns a raw entry-point call into one or more normalized interactions."""

    @abstractmethod
    def matches(self, raw: RawCall) -> bool: ...

    @abstractmethod
    def normalize(self, raw: RawCall, ctx: SessionContext) -> list[Interaction]: ...


class Control(ABC):
    """A guardrail evaluated on interactions at the stages it declares."""

    id: ClassVar[str]
    stages: ClassVar[frozenset[Stage]]  # a control can run pre, post, or both
    kind: ClassVar[ControlKind]
    mandatory: ClassVar[bool] = False  # enforces in every profile, cannot be log_only
    # A sealing pre control (e.g. sql_guard) runs once, after every other pre control's rewrites
    # and redactions, on the final payload, and nothing may change the payload after it allows
    # the call: the pipeline refuses a dispatched payload that differs with this reason code.
    # A sealing control may rewrite, never redact. None = an ordinary control.
    seal: ClassVar[str | None] = None

    @abstractmethod
    async def evaluate(self, interaction: Interaction, stage: Stage, cfg: ControlConfig) -> Verdict:
        """Return this control's verdict; a log_only control returns ``enforced=False``."""
