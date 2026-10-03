"""``model_allowlist`` on its own: the model sent is authorized, the model that answered matches."""

from typing import Any

import pytest

from gateway.controls.model_allowlist import ModelAllowlistControl, accepted_names
from gateway.controls.scope import CallScope, call_scope
from gateway.core.envelope import Interaction
from gateway.core.interfaces import ControlConfig
from gateway.core.types import Action, Channel, ControlMode, Decision, SessionMode, Stage
from gateway.policy.evaluator import PolicyEvaluator, PrincipalContext
from gateway.policy.schema import ModelAllowlistConfig

BLOCK = ModelAllowlistConfig(mode=ControlMode.BLOCK)
LOG_ONLY = ModelAllowlistConfig(mode=ControlMode.LOG_ONLY)
ROLES = {"anna@demo": ("analyst",), "bartek@demo": ("intern",)}


@pytest.fixture
def control() -> ModelAllowlistControl:
    return ModelAllowlistControl(PolicyEvaluator())


@pytest.fixture
def call(make_ctx, snapshot):
    """Evaluate a chat interaction under the call scope of ``principal``."""

    async def run(
        stage: Stage,
        *,
        principal: str = "anna@demo",
        requested: str = "qwen3:8b",
        sent: str | None = None,
        answered: str | None = None,
        cfg: ControlConfig = BLOCK,
        scoped: bool = True,
        control: ModelAllowlistControl,
    ):
        payload: dict[str, Any] = {"model": sent or requested, "messages": []}
        result = None if answered is None else {"model": answered, "choices": []}
        interaction = Interaction(
            session_id="s-test",
            principal=principal,
            actor="databot",
            mode=SessionMode.INTERACTIVE,
            channel=Channel.LLM,
            action=Action.GENERATE,
            resource=f"model:{requested}",
            payload=payload,
            result=result,
            context=make_ctx(principal=principal),
        )
        if not scoped:
            return await control.evaluate(interaction, stage, cfg)
        scope = CallScope(
            snapshot=snapshot,
            principal=PrincipalContext(
                principal=principal,
                roles=ROLES[principal],
                agent="databot",
                mode=SessionMode.INTERACTIVE,
            ),
        )
        with call_scope(scope):
            return await control.evaluate(interaction, stage, cfg)

    return run


@pytest.mark.parametrize("cfg", [BLOCK, LOG_ONLY], ids=["block", "log_only"])
@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        pytest.param({"principal": "anna@demo", "requested": "llama3:70b"}, None, id="analyst-any"),
        pytest.param({"principal": "bartek@demo"}, None, id="intern-own-model"),
        pytest.param(
            {"principal": "bartek@demo", "requested": "llama3:70b"},
            "model_not_allowed",
            id="intern-other-model",
        ),
        pytest.param({"sent": "llama3:70b"}, "model_not_allowed", id="payload-differs"),
        pytest.param({"sent": "bad model"}, "model_not_allowed", id="invalid-model"),
        pytest.param({"scoped": False}, "model_unverifiable", id="no-call-scope"),
    ],
)
async def test_pre(control, call, kwargs, reason, cfg):
    verdict = await call(Stage.PRE, control=control, cfg=cfg, **kwargs)
    if reason is None:
        assert (verdict.decision, verdict.reason_code) == (Decision.ALLOW, "model_allowed")
        return
    assert (verdict.decision, verdict.reason_code) == (Decision.BLOCK, reason)
    assert verdict.enforced is (cfg.mode is ControlMode.BLOCK)


@pytest.mark.parametrize("cfg", [BLOCK, LOG_ONLY], ids=["block", "log_only"])
@pytest.mark.parametrize(
    ("requested", "answered", "allowed"),
    [
        pytest.param("qwen3:8b", "qwen3:8b", True, id="same"),
        pytest.param("qwen3", "qwen3:latest", True, id="implicit-latest"),
        pytest.param("qwen3:8b", "qwen3:latest", False, id="tagged-is-exact"),
        pytest.param("qwen3:8b", "llama3:70b", False, id="substituted"),
        pytest.param("qwen3:8b", "", False, id="missing"),
    ],
)
async def test_post(control, call, requested, answered, allowed, cfg):
    verdict = await call(
        Stage.POST, control=control, cfg=cfg, requested=requested, answered=answered or ""
    )
    if allowed:
        assert (verdict.decision, verdict.reason_code) == (Decision.ALLOW, "model_matches")
        return
    assert (verdict.decision, verdict.reason_code) == (Decision.BLOCK, "model_mismatch")
    assert verdict.enforced is (cfg.mode is ControlMode.BLOCK)


async def test_post_accepts_a_configured_alias(control, call):
    cfg = ModelAllowlistConfig.model_validate({"mode": "block", "aliases": {"fast": ["qwen3:8b"]}})
    verdict = await call(
        Stage.POST, control=control, cfg=cfg, requested="fast", answered="qwen3:8b"
    )
    assert verdict.decision is Decision.ALLOW
    blocked = await call(Stage.POST, control=control, requested="fast", answered="qwen3:8b")
    assert blocked.reason_code == "model_mismatch"  # without the alias


def test_accepted_names():
    assert accepted_names("qwen3", ControlConfig()) == {"qwen3", "qwen3:latest"}
    assert accepted_names("qwen3:8b", ControlConfig()) == {"qwen3:8b"}
