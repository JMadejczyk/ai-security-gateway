"""``model_allowlist``: the model a call is sent to, and the model that answers, are allowed.

Pre (defense in depth on top of ``authz``): the model in the payload that is about to be sent
upstream must be the model the call was authorized for, and ``generate:model:<model>`` must
pass the evaluator again under the call's own policy snapshot and principal. ``authz`` checked
the resource the adapter derived; this checks the bytes that leave, after every earlier control
had its chance to rewrite them.

Post: the upstream's own ``model`` field must name the model that was requested, a configured
alias of it (``controls.model_allowlist.aliases``), or, for an untagged name, its ``:latest``
tag (Ollama reports ``qwen3`` as ``qwen3:latest``). A router that silently substitutes another
model is caught here (``model_mismatch``). A missing ``model`` field counts as a mismatch.
"""

from collections.abc import Mapping
from typing import Any, ClassVar, Final, cast, override

from gateway.controls.scope import current_scope
from gateway.core.envelope import Interaction, Verdict
from gateway.core.interfaces import Control, ControlConfig
from gateway.core.types import Action, ControlKind, ControlMode, Decision, Stage
from gateway.policy.evaluator import PolicyEvaluator
from gateway.policy.permissions import Resource
from gateway.policy.schema import ModelAllowlistConfig

MODEL_NOT_ALLOWED: Final = "model_not_allowed"
MODEL_UNVERIFIABLE: Final = "model_unverifiable"
MODEL_MISMATCH: Final = "model_mismatch"
LATEST_TAG: Final = ":latest"


def model_of(document: object) -> str | None:
    """The ``model`` field of a chat request or completion, if it is a non-empty string."""
    if not isinstance(document, Mapping):
        return None
    model = cast("Mapping[str, Any]", document).get("model")
    return model if isinstance(model, str) and model else None


def accepted_names(requested: str, cfg: ControlConfig) -> frozenset[str]:
    """Names the upstream may report for ``requested``."""
    aliases = cfg.aliases.get(requested, ()) if isinstance(cfg, ModelAllowlistConfig) else ()
    implicit = (f"{requested}{LATEST_TAG}",) if ":" not in requested else ()
    return frozenset({requested, *aliases, *implicit})


class ModelAllowlistControl(Control):
    id: ClassVar[str] = "model_allowlist"
    stages: ClassVar[frozenset[Stage]] = frozenset({Stage.PRE, Stage.POST})
    kind: ClassVar[ControlKind] = ControlKind.DETERMINISTIC

    def __init__(self, evaluator: PolicyEvaluator) -> None:
        self._evaluator = evaluator

    @override
    async def evaluate(self, interaction: Interaction, stage: Stage, cfg: ControlConfig) -> Verdict:
        sent = model_of(interaction.payload)
        if stage is Stage.PRE:
            return self._before(interaction, sent, cfg)
        return self._after(interaction, sent, cfg)

    def _before(self, interaction: Interaction, sent: str | None, cfg: ControlConfig) -> Verdict:
        scope = current_scope()
        if sent is None or scope is None:
            return self._refuse(MODEL_UNVERIFIABLE, "no model or call scope to check", cfg)
        try:
            resource = str(Resource.parse(f"model:{sent}"))
        except ValueError:
            return self._refuse(MODEL_NOT_ALLOWED, "the model is not a valid identifier", cfg)
        if resource != interaction.resource:
            return self._refuse(MODEL_NOT_ALLOWED, "the payload names another model", cfg)
        authorization = self._evaluator.authorize(
            scope.snapshot, scope.principal, Action.GENERATE, resource
        )
        if not authorization.allowed:
            return self._refuse(MODEL_NOT_ALLOWED, str(authorization.reason_code), cfg)
        return Verdict(decision=Decision.ALLOW, control_id=self.id, reason_code="model_allowed")

    def _after(self, interaction: Interaction, sent: str | None, cfg: ControlConfig) -> Verdict:
        answered = model_of(interaction.result)
        if sent is not None and answered in accepted_names(sent, cfg):
            return Verdict(decision=Decision.ALLOW, control_id=self.id, reason_code="model_matches")
        return self._refuse(MODEL_MISMATCH, "the upstream answered with another model", cfg)

    def _refuse(self, reason_code: str, reason: str, cfg: ControlConfig) -> Verdict:
        return Verdict(
            decision=Decision.BLOCK,
            control_id=self.id,
            reason_code=reason_code,
            reason=reason,
            enforced=cfg.mode is not ControlMode.LOG_ONLY,
            risk_delta=cfg.risk_delta or 0.0,
        )
