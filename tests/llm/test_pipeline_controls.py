"""The control extension point: verdicts merge, rewrite, redact, log_only, fail closed, taint.

Real controls arrive in stages 5-10; these scripted stand-ins prove the pipeline seams they
will plug into, through the HTTP app.
"""

import json
from collections.abc import Callable
from typing import ClassVar

import httpx
import pytest
from gateway_testkit import bearer, chat, completion

from gateway.controls.registry import ControlRegistry
from gateway.core.envelope import Interaction, Span, Verdict
from gateway.core.interfaces import Control, ControlConfig
from gateway.core.types import Channel, ControlKind, Decision, Stage
from gateway.telemetry import REGISTRY

CHAT = "/v1/chat/completions"
type Behaviour = Callable[[Interaction, Stage, ControlConfig], Verdict]


def allow(control_id: str) -> Verdict:
    return Verdict(decision=Decision.ALLOW, control_id=control_id, reason_code="clean")


class ScriptedPii(Control):
    id: ClassVar[str] = "pii"
    stages: ClassVar[frozenset[Stage]] = frozenset({Stage.PRE, Stage.POST})
    kind: ClassVar[ControlKind] = ControlKind.DETERMINISTIC

    def __init__(self, behave: Behaviour) -> None:
        self.behave = behave
        self.seen: list[tuple[Stage, ControlConfig]] = []

    async def evaluate(self, interaction, stage, cfg):
        self.seen.append((stage, cfg))
        return self.behave(interaction, stage, cfg)


class ScriptedSecrets(ScriptedPii):
    id: ClassVar[str] = "secrets"
    mandatory: ClassVar[bool] = True


class ScriptedInjection(ScriptedPii):
    id: ClassVar[str] = "prompt_injection"
    kind: ClassVar[ControlKind] = ControlKind.SEMANTIC


@pytest.fixture
def upstream(llm_upstream):
    return llm_upstream.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=completion("Anna's PESEL is 44051401359."))
    )


async def ask(gateway, sub: str = "anna@demo", **body):
    token = await gateway.token(sub)
    return await gateway.agent.post(CHAT, json=chat(**body), headers=bearer(token))


async def test_post_redaction_reaches_json_and_sse(gateway, upstream):
    def redact_pesel(interaction, stage, cfg):
        if stage is Stage.PRE:
            return allow("pii")
        text = interaction.result["choices"][0]["message"]["content"]
        start = text.index("44051401359")
        span = Span(path="/choices/0/message/content", start=start, end=start + 11, label="PESEL")
        return Verdict(
            decision=Decision.REDACT,
            control_id="pii",
            reason_code="pii_detected",
            risk_delta=cfg.risk_delta or 0.0,
            redactions=(span,),
        )

    control = ScriptedPii(redact_pesel)
    gateway.container.pipeline.controls.register(control)
    response = await ask(gateway)
    assert response.status_code == 200
    assert (
        response.json()["choices"][0]["message"]["content"] == "Anna's PESEL is [REDACTED:PESEL]."
    )
    streamed = await ask(gateway, stream=True)
    assert "44051401359" not in streamed.text
    assert "[REDACTED:PESEL]" in streamed.text
    # The control received its resolved config: profile mode and catalog risk delta.
    stage, cfg = control.seen[-1]
    assert (stage, cfg.mode, cfg.risk_delta) == (Stage.POST, "redact", 0.1)
    entry = gateway.audit_entries()[0]
    assert entry["decision"] == "redact"
    assert {"control": "pii", "stage": "post", "decision": "redact", "enforced": True,
            "reason_code": "pii_detected"} in entry["verdicts"]  # fmt: skip


async def test_pre_block_stops_before_upstream_and_adds_risk(gateway, upstream):
    def block_secret(interaction, stage, cfg):
        return Verdict(
            decision=Decision.BLOCK,
            control_id="secrets",
            reason_code="secret_detected",
            risk_delta=cfg.risk_delta or 0.0,
        )

    gateway.container.pipeline.controls.register(ScriptedSecrets(block_secret))
    response = await ask(gateway)
    assert (response.status_code, response.json()["error"]["code"]) == (403, "secret_detected")
    assert not upstream.called
    (entry,) = gateway.audit_entries()
    assert entry["risk"] == pytest.approx(0.3)


async def test_pre_rewrite_is_what_the_upstream_executes(gateway, upstream):
    def rewrite(interaction, stage, cfg):
        if stage is Stage.POST:
            return allow("pii")
        payload = {**interaction.payload, "messages": [{"role": "user", "content": "[scrubbed]"}]}
        return Verdict(
            decision=Decision.ALLOW, control_id="pii", reason_code="rewritten", rewrite=payload
        )

    gateway.container.pipeline.controls.register(ScriptedPii(rewrite))
    await ask(gateway)
    sent = json.loads(upstream.calls.last.request.content)
    assert sent["messages"] == [{"role": "user", "content": "[scrubbed]"}]


async def test_log_only_is_recorded_but_not_applied(gateway, upstream):
    text = gateway.policy_path.read_text().replace(
        "pii:              { mode: redact,", "pii:              { mode: log_only,", 1
    )
    gateway.policy_path.write_text(text)
    assert gateway.container.policy_store.reload().error is None
    gateway.container.pipeline.controls.register(
        ScriptedPii(
            lambda i, s, c: Verdict(
                decision=Decision.BLOCK, control_id="pii", reason_code="pii_detected"
            )
        )
    )
    response = await ask(gateway)
    assert response.status_code == 200
    verdicts = gateway.audit_entries()[0]["verdicts"]
    assert {"control": "pii", "stage": "pre", "decision": "block", "enforced": False,
            "reason_code": "pii_detected"} in verdicts  # fmt: skip


async def test_a_crashing_control_fails_closed(gateway, upstream):
    def crash(interaction, stage, cfg):
        raise RuntimeError("bug")

    gateway.container.pipeline.controls.register(ScriptedSecrets(crash))
    response = await ask(gateway)
    assert (response.status_code, response.json()["error"]["code"]) == (403, "control_error")
    assert not upstream.called


async def test_post_block_withholds_the_answer(gateway, upstream):
    def block_post(interaction, stage, cfg):
        if stage is Stage.PRE:
            return allow("secrets")
        return Verdict(
            decision=Decision.BLOCK, control_id="secrets", reason_code="secret_in_output"
        )

    gateway.container.pipeline.controls.register(ScriptedSecrets(block_post))
    response = await ask(gateway)
    assert (response.status_code, response.json()["error"]["code"]) == (403, "secret_in_output")
    assert "44051401359" not in response.text
    assert upstream.call_count == 1


async def test_injection_taints_the_session(gateway, upstream):
    tainted_before = REGISTRY.get_sample_value("acl_tainted_sessions", {})

    def detect(interaction, stage, cfg):
        return Verdict(
            decision=Decision.BLOCK, control_id="prompt_injection", reason_code="injection",
            risk_delta=cfg.risk_delta or 0.0,
        )  # fmt: skip

    gateway.container.pipeline.controls.register(ScriptedInjection(detect))
    await ask(gateway)
    (entry,) = gateway.audit_entries()
    assert (entry["taint"], entry["risk"]) == (True, pytest.approx(0.6))
    assert REGISTRY.get_sample_value("acl_tainted_sessions", {}) == (tainted_before or 0) + 1


async def test_require_approval_is_held(gateway, upstream):
    gateway.container.pipeline.controls.register(
        ScriptedPii(
            lambda i, s, c: Verdict(
                decision=Decision.REQUIRE_APPROVAL, control_id="pii", reason_code="needs_review"
            )
        )
    )
    response = await ask(gateway)
    assert (response.status_code, response.json()["error"]["code"]) == (403, "approval_required")
    assert not upstream.called


def test_registry_rejects_pipeline_steps_and_catalog_mismatches():
    class FakeAuthz(ScriptedPii):
        id: ClassVar[str] = "authz"

    class SemanticPii(ScriptedPii):
        kind: ClassVar[ControlKind] = ControlKind.SEMANTIC

    class Unknown(ScriptedPii):
        id: ClassVar[str] = "telepathy"

    registry = ControlRegistry()
    with pytest.raises(ValueError, match="pipeline step"):
        registry.register(FakeAuthz(lambda i, s, c: allow("authz")))
    with pytest.raises(ValueError, match="catalog"):
        registry.register(SemanticPii(lambda i, s, c: allow("pii")))
    with pytest.raises(KeyError):
        registry.register(Unknown(lambda i, s, c: allow("telepathy")))
    registry.register(ScriptedInjection(lambda i, s, c: allow("prompt_injection")))
    registry.register(ScriptedPii(lambda i, s, c: allow("pii")))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(ScriptedPii(lambda i, s, c: allow("pii")))
    # Deterministic controls run before semantic ones, whatever the registration order.
    assert [c.id for c in registry.for_stage(Stage.PRE, Channel.LLM)] == ["pii", "prompt_injection"]
