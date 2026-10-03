"""Review findings on the shared pipeline, exercised through the LLM entry point.

Rewrites are re-authorized, throttles are enforced with backoff and alerts, retired sessions
never come back clean, compressed upstream bodies are refused, control crashes log no payload,
the token metric's model label is the authorized model, and early refusals are audited.
"""

import gzip
import json
import logging
from typing import ClassVar

import httpx
import jwt
import pytest
from gateway_testkit import bearer, chat, claims, completion, sign

from gateway.core.envelope import Verdict
from gateway.core.interfaces import Control
from gateway.core.types import ControlKind, Decision, Stage
from gateway.sessions import SessionUpdate
from gateway.telemetry import REGISTRY

CHAT = "/v1/chat/completions"
ETL = "svc:nightly_etl"


def sample(name: str, labels: dict[str, str]) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


class Scripted(Control):
    """A deterministic pre/post control whose behaviour each test scripts."""

    id: ClassVar[str] = "pii"
    stages: ClassVar[frozenset[Stage]] = frozenset({Stage.PRE, Stage.POST})
    kind: ClassVar[ControlKind] = ControlKind.DETERMINISTIC

    def __init__(self, behave) -> None:
        self.behave = behave

    async def evaluate(self, interaction, stage, cfg):
        return self.behave(interaction, stage)


def allow(reason: str = "clean", rewrite: object = None) -> Verdict:
    return Verdict(decision=Decision.ALLOW, control_id="pii", reason_code=reason, rewrite=rewrite)


@pytest.fixture
def upstream(llm_upstream):
    return llm_upstream.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=completion())
    )


async def raise_risk(gateway, token: str, delta: float) -> str:
    """Open the token's session and push its risk up directly; returns the session id."""
    session_id = jwt.decode(token, options={"verify_signature": False})["session_id"]
    await gateway.container.sessions.apply(
        session_id, SessionUpdate(risk_delta=delta), half_life_s=600
    )
    return session_id


# ----------------------------------------------------------- 1. rewrites are authorized


async def test_rewrite_to_an_unauthorized_model_is_blocked(gateway, upstream):
    def to_llama(interaction, stage):
        if stage is Stage.POST:
            return allow()
        return allow("rewritten", rewrite={**interaction.payload, "model": "llama3:70b"})

    gateway.container.pipeline.controls.clear()  # scripted below, replacing the real controls

    gateway.container.pipeline.controls.register(Scripted(to_llama))
    response = await gateway.agent.post(
        CHAT, json=chat(), headers=bearer(await gateway.token("bartek@demo"))
    )
    assert (response.status_code, response.json()["error"]["code"]) == (
        403,
        "rewrite_unauthorized",
    )
    assert not upstream.called
    entry = gateway.audit_entries()[-1]
    assert (entry["resource"], entry["decision"]) == ("model:llama3:70b", "block")
    assert entry["risk"] == pytest.approx(0.1)  # a refused rewrite is an authz deny


async def test_rewrite_into_an_invalid_request_is_blocked(gateway, upstream):
    gateway.container.pipeline.controls.clear()  # scripted below, replacing the real controls
    gateway.container.pipeline.controls.register(
        Scripted(lambda i, s: allow(rewrite={"model": "qwen3:8b"}) if s is Stage.PRE else allow())
    )
    response = await gateway.agent.post(
        CHAT, json=chat(), headers=bearer(await gateway.token("anna@demo"))
    )
    assert response.json()["error"]["code"] == "rewrite_unauthorized"
    assert not upstream.called


async def test_authorized_rewrite_still_runs(gateway, upstream):
    def lower_temperature(interaction, stage):
        if stage is Stage.POST:
            return allow()
        return allow(rewrite={**interaction.payload, "temperature": 0.0})

    gateway.container.pipeline.controls.clear()  # scripted below, replacing the real controls

    gateway.container.pipeline.controls.register(Scripted(lower_temperature))
    response = await gateway.agent.post(
        CHAT, json=chat(temperature=1.5), headers=bearer(await gateway.token("bartek@demo"))
    )
    assert response.status_code == 200
    assert json.loads(upstream.calls.last.request.content)["temperature"] == 0.0


# ---------------------------------------------------------------- 2. throttling + alerts


async def test_autonomous_agent_at_elevated_risk_is_throttled(gateway, upstream, caplog):
    token = await gateway.token(ETL)
    first = await gateway.agent.post(CHAT, json=chat(), headers=bearer(token))
    assert first.status_code == 200  # risk 0: no throttle yet
    await raise_risk(gateway, token, 0.6)  # > 0.5: max 1 call per 10 s, alert
    throttled_before = sample("acl_throttled_total", {"agent": "nightly_etl"})
    alerts_before = sample("acl_alerts_total", {"rule": "autonomous.1"})

    with caplog.at_level(logging.WARNING, logger="gateway.alerts"):
        codes = [
            (await gateway.agent.post(CHAT, json=chat(), headers=bearer(token))) for _ in range(3)
        ]
    assert [r.status_code for r in codes] == [200, 429, 429]
    assert [r.headers.get("retry-after") for r in codes] == [None, "5", "10"]  # base_s, doubled
    assert codes[1].json()["error"] == {
        "message": "rate limited; retry after 5 s",
        "type": "rate_limit_error",
        "code": "throttled",
    }
    assert upstream.call_count == 2
    assert sample("acl_throttled_total", {"agent": "nightly_etl"}) == throttled_before + 2
    assert sample("acl_alerts_total", {"rule": "autonomous.1"}) == alerts_before + 3
    alert = json.loads(next(r.getMessage() for r in caplog.records if r.name == "gateway.alerts"))
    assert (alert["event"], alert["rule"], alert["agent"]) == (
        "risk_rule_alert",
        "autonomous.1",
        "nightly_etl",
    )
    assert gateway.audit_entries()[-1]["reason_code"] == "throttled"


async def test_throttle_backoff_resets_after_a_compliant_window(gateway, upstream):
    token = await gateway.token(ETL)
    await gateway.agent.post(CHAT, json=chat(), headers=bearer(token))
    await raise_risk(gateway, token, 0.6)

    prompts = iter(range(100))

    async def call() -> httpx.Response:  # distinct prompts: identical ones trip loop_detect
        body = chat(messages=[{"role": "user", "content": f"step {next(prompts)}"}])
        return await gateway.agent.post(CHAT, json=body, headers=bearer(token))

    assert (await call()).status_code == 200
    assert (await call()).headers["retry-after"] == "5"
    gateway.clock.advance(3)  # still backing off: another rejection, doubled
    assert (await call()).headers["retry-after"] == "10"
    gateway.clock.advance(10 + 10)  # backoff over, then one full window without a violation
    assert (await call()).status_code == 200
    assert (await call()).headers["retry-after"] == "5"  # back to base_s


async def test_interactive_sessions_are_never_throttled(gateway, upstream):
    token = await gateway.token("anna@demo")
    await gateway.agent.post(CHAT, json=chat(), headers=bearer(token))
    await raise_risk(gateway, token, 0.6)
    codes = [
        (await gateway.agent.post(CHAT, json=chat(), headers=bearer(token))).status_code
        for _ in range(3)
    ]
    assert codes == [200, 200, 200]


# ------------------------------------------------------- 3. retired sessions stay retired


@pytest.mark.parametrize("how", ["deleted", "idle"])
async def test_retired_tainted_session_never_comes_back_clean(gateway, upstream, how):
    token = sign(claims(gateway.clock, session_id="s-tainted"))
    await gateway.agent.post(CHAT, json=chat(), headers=bearer(token))
    await gateway.container.sessions.apply(
        "s-tainted", SessionUpdate(risk_delta=0.4, taint=True), half_life_s=600
    )
    if how == "deleted":
        assert (await gateway.agent.delete("/v1/session", headers=bearer(token))).status_code == 200
    else:
        gateway.clock.advance(3600)  # sessions.idle_ttl_s
    gateway.clock.advance(86400 + 3600 + 1)  # past max_lifetime_s and any token lifetime

    # Signed with the key but naming the old id with a new creation time: refused by the store.
    late = sign(claims(gateway.clock, session_id="s-tainted"))
    response = await gateway.agent.post(CHAT, json=chat(), headers=bearer(late))
    assert (response.status_code, response.json()["error"]["code"]) == (401, "session_ended")
    assert await gateway.container.sessions.get("s-tainted") is None
    # The demo issuer never lets a caller name a session at all.
    reissue = await gateway.operator.post(
        "/auth/demo-token", json={"sub": "anna@demo", "session_id": "s-tainted"}
    )
    assert reissue.status_code == 422
    assert upstream.call_count == 1


# --------------------------------------------------------- 4. compressed upstream bodies


async def test_gzip_bomb_from_the_upstream_is_refused(gateway, llm_upstream):
    body = json.dumps(completion("a" * 8_000_000)).encode()  # inflates far past the 4 MiB cap
    route = llm_upstream.post("/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=gzip.compress(body),
            headers={"content-type": "application/json", "content-encoding": "gzip"},
        )
    )
    response = await gateway.agent.post(
        CHAT, json=chat(), headers=bearer(await gateway.token("anna@demo"))
    )
    assert (response.status_code, response.json()["error"]["code"]) == (
        502,
        "upstream_encoding_refused",
    )
    assert route.calls.last.request.headers["accept-encoding"] == "identity"


# ------------------------------------------------------------ 6. no payload in the logs


async def test_a_crashing_control_logs_no_payload(gateway, upstream, caplog):
    sensitive = "sk-live-4f9a-the-merger-plan"

    def crash(interaction, stage):
        raise ValueError(f"cannot parse {interaction.payload['messages'][0]['content']}")

    gateway.container.pipeline.controls.clear()  # scripted below, replacing the real controls

    gateway.container.pipeline.controls.register(Scripted(crash))
    token = await gateway.token("anna@demo")
    body = chat(messages=[{"role": "user", "content": sensitive}])
    with caplog.at_level(logging.DEBUG):
        response = await gateway.agent.post(CHAT, json=body, headers=bearer(token))
    assert response.json()["error"]["code"] == "control_error"
    logged = "\n".join(
        [caplog.text, *(caplog.handler.format(r) for r in caplog.records)]  # incl. tracebacks
    )
    assert sensitive not in logged
    assert "merger" not in logged
    (record,) = [r for r in caplog.records if "control_error" in r.getMessage()]
    assert record.exc_info is None
    assert "control=pii" in record.getMessage()
    assert "error=ValueError" in record.getMessage()


# ------------------------------------------------------------- 7. bounded model label


async def test_token_metric_uses_the_authorized_model_not_the_upstream_string(
    gateway, llm_upstream
):
    llm_upstream.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=completion(model="attacker-chosen-label-8f3a"))
    )
    labels = {"user": "anna@demo", "agent": "databot", "model": "qwen3:8b"}
    before = sample("acl_tokens_total", labels)
    await gateway.agent.post(CHAT, json=chat(), headers=bearer(await gateway.token("anna@demo")))
    assert sample("acl_tokens_total", labels) == before + 19
    exposed = (await gateway.operator.get("/metrics")).text
    assert "attacker-chosen-label-8f3a" not in exposed


# ------------------------------------------------------- 8. early refusals are audited


async def test_early_413_is_audited_with_identity(gateway, upstream):
    blocked = {"channel": "llm", "decision": "block", "agent": "databot"}
    before = sample("acl_requests_total", blocked)
    token = await gateway.token("bartek@demo")
    huge = chat(messages=[{"role": "user", "content": "x" * 1_100_000}])
    response = await gateway.agent.post(CHAT, json=huge, headers=bearer(token))
    assert response.status_code == 413
    (entry,) = gateway.audit_entries()
    assert (entry["reason_code"], entry["status"], entry["principal"], entry["actor"]) == (
        "request_too_large",
        413,
        "bartek@demo",
        "databot",
    )
    assert entry["payload_hmac"] is None
    assert sample("acl_requests_total", blocked) == before + 1
    assert not upstream.called


async def test_early_413_without_a_valid_token_is_audited_anonymously(gateway, upstream):
    huge = chat(messages=[{"role": "user", "content": "x" * 1_100_000}])
    response = await gateway.agent.post(CHAT, json=huge, headers=bearer("forged"))
    assert response.status_code == 413
    (entry,) = gateway.audit_entries()
    assert (entry["reason_code"], entry["principal"]) == ("request_too_large", None)
