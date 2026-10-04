"""The opt-in remote LLM upstream (``ACL_LLM_UPSTREAM=remote``, OpenRouter with ZDR routing).

respx stands in for OpenRouter. Inside the gateway every model keeps its logical id; the
proxy maps it to the provider id on the way out and back on the way in, merges the policy's
``extra_body`` into every request (agent and judge), translates ``reasoning_effort``, and
sends only the gateway's key. Pre controls still run before anything leaves, and the local
upstream is untouched when remote is not selected.
"""

import json
import logging
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
import yaml
from gateway_testkit import (
    ROOT_POLICY,
    T0,
    Harness,
    MutableClock,
    bearer,
    chat,
    completion,
    make_settings,
    running_gateway,
)
from injection_kit import INJECT_MARKER
from pydantic import BaseModel, ConfigDict

from gateway.budget.ledger import BudgetLedger
from gateway.budget.model import BudgetedCall, Meter
from gateway.budget.store import InMemoryBudgetStore
from gateway.container import GatewayContainer
from gateway.core.types import Channel, LlmUpstreamKind
from gateway.errors import StartupError
from gateway.judges.client import JudgeClient
from gateway.policy.loader import PolicyLoader, PolicyLoadError
from gateway.proxies.llm import outgoing_request, require_upstream
from gateway.telemetry import REGISTRY, ReloadResult
from gateway.upstream import TokenUsage, UpstreamResult

CHAT = "/v1/chat/completions"
REMOTE = "https://openrouter.ai/api/v1"
KEY_ENV = "OPENROUTER_API_KEY"
KEY = "remote-test-key-" + "q8Zr2Lm5Tx9Wc4Vb7Nd1"  # fake, assembled
LOGICAL = "deepseek-v4.1-flash"  # its own logical id: qwen3:8b stays local-only
PROVIDER = "deepseek/deepseek-v4.1-flash"
ZDR_TERMS = {
    "provider": {
        "zdr": True,
        "data_collection": "deny",
        "max_price": {"prompt": 0.3, "completion": 1.2},
    }
}
PESEL = "44051401359"
API_KEY = "sk-proj-" + "Ab3De5Fg7Hi9Jk1Lm3No5Pq7Rs9Tu1Vw3Xy5Za7Bc9De"  # fake, assembled
ZDR_REFUSAL = "No endpoints found matching your data policy (Zero data retention). zdr-detail"

ALLOWLIST_ALLOW = pytest.mark.control("model_allowlist", "allow")
ALLOWLIST_DENY = pytest.mark.control("model_allowlist", "deny")


@pytest.fixture
async def remote(tmp_path: Path) -> AsyncIterator[Harness]:
    async with running_gateway(tmp_path, llm_upstream="remote", env={KEY_ENV: KEY}) as harness:
        yield harness


@pytest.fixture
def openrouter() -> Iterator[respx.MockRouter]:
    with respx.mock(base_url=REMOTE, assert_all_called=False) as router:
        yield router


@pytest.fixture
def answers(openrouter: respx.MockRouter) -> respx.Route:
    return openrouter.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=completion(model=PROVIDER))
    )


async def ask(gateway: Harness, sub: str = "anna@demo", **body: Any) -> httpx.Response:
    token = await gateway.token(sub)
    request = chat(**{"model": LOGICAL, **body})
    return await gateway.agent.post(CHAT, json=request, headers=bearer(token))


def sent(route: respx.Route) -> dict[str, Any]:
    return json.loads(route.calls.last.request.content)


# --------------------------------------------------------------------- model mapping


@ALLOWLIST_ALLOW
async def test_the_model_is_mapped_out_and_back(remote, answers):
    response = await ask(remote)
    assert response.status_code == 200
    assert sent(answers)["model"] == PROVIDER
    assert response.json()["model"] == LOGICAL  # the agent only ever sees logical ids
    (entry,) = remote.audit_entries()
    assert entry["resource"] == f"model:{LOGICAL}"
    assert {"control": "model_allowlist", "stage": "post", "decision": "allow"}.items() <= next(
        v for v in entry["verdicts"] if (v["control"], v["stage"]) == ("model_allowlist", "post")
    ).items()


@ALLOWLIST_DENY
async def test_an_answer_from_an_unmapped_provider_model_is_a_mismatch(remote, openrouter):
    openrouter.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=completion(model="deepseek/deepseek-v4-flash"))
    )
    response = await ask(remote)
    assert (response.status_code, response.json()["error"]["code"]) == (403, "model_mismatch")


async def test_a_local_model_is_refused_in_remote_mode_before_egress(remote, answers):
    response = await ask(remote, model="qwen3:8b")  # anna may generate it; remote serves it not
    error = response.json()["error"]
    assert (response.status_code, error["code"]) == (400, "model_not_mapped")
    assert error["message"] == f"qwen3:8b is local-only; the remote upstream serves: {LOGICAL}"
    assert not answers.called
    (entry,) = remote.audit_entries()
    assert (entry["reason_code"], entry["upstream"]) == ("model_not_mapped", "remote")


async def test_models_lists_the_mapped_logical_ids_only(remote, openrouter):
    listing = openrouter.get("/models")
    for sub in ("anna@demo", "bartek@demo", "svc:nightly_etl"):  # each holds the grant
        token = await remote.token(sub)
        response = await remote.agent.get("/v1/models", headers=bearer(token))
        assert [m["id"] for m in response.json()["data"]] == [LOGICAL], sub
    olga = await remote.token("olga@demo")  # ops-team grants nothing
    assert (await remote.agent.get("/v1/models", headers=bearer(olga))).json()["data"] == []
    assert not listing.called  # nothing to ask: the policy names the models


async def test_a_remote_model_is_refused_in_local_mode(gateway, llm_upstream):
    route = llm_upstream.post("/chat/completions")
    token = await gateway.token("anna@demo")
    response = await gateway.agent.post(CHAT, json=chat(LOGICAL), headers=bearer(token))
    error = response.json()["error"]
    assert (response.status_code, error["code"]) == (400, "model_not_mapped")
    assert error["message"] == f"{LOGICAL} is served by the remote upstream only (make remote-up)"
    assert not route.called


# ------------------------------------------------------------ what leaves the machine


async def test_extra_body_wins_and_router_extensions_from_the_agent_are_dropped(remote, answers):
    response = await ask(
        remote,
        provider={"zdr": False, "data_collection": "allow"},
        models=["openai/gpt-4o", "anthropic/claude-3"],
        plugins=[{"id": "web"}],
        transforms=["middle-out"],
        user="anna@demo",
        temperature=0.2,
    )
    assert response.status_code == 200
    body = sent(answers)
    assert body["provider"] == ZDR_TERMS["provider"]
    assert not {"models", "plugins", "transforms", "user"} & set(body)
    assert (body["temperature"], body["stream"]) == (0.2, False)


@pytest.mark.parametrize(
    ("effort", "translated"),
    [("none", {"enabled": False}), ("low", {"effort": "low"}), ("high", {"effort": "high"})],
)
async def test_reasoning_effort_is_translated_for_this_upstream(
    remote, answers, effort, translated
):
    await ask(remote, reasoning_effort=effort)
    body = sent(answers)
    assert body["reasoning"] == translated
    assert "reasoning_effort" not in body


async def test_only_the_gateway_key_reaches_the_remote_upstream(remote, answers):
    token = await remote.token("anna@demo")
    await remote.agent.post(CHAT, json=chat(LOGICAL), headers=bearer(token))
    headers = answers.calls.last.request.headers
    assert headers["authorization"] == f"Bearer {KEY}"
    assert token not in str(headers)
    assert headers["accept-encoding"] == "identity"
    assert KEY not in remote.audit.getvalue()


async def test_a_zdr_routing_refusal_is_a_generic_502(remote, openrouter):
    openrouter.post("/chat/completions").mock(
        return_value=httpx.Response(404, json={"error": {"code": 404, "message": ZDR_REFUSAL}})
    )
    response = await ask(remote)
    assert (response.status_code, response.json()["error"]["code"]) == (502, "upstream_error")
    assert "zdr-detail" not in response.text
    assert "zdr-detail" not in remote.audit.getvalue()


async def test_an_error_object_with_status_200_is_a_generic_502(remote, openrouter):
    openrouter.post("/chat/completions").mock(
        return_value=httpx.Response(200, json={"error": {"code": 502, "message": ZDR_REFUSAL}})
    )
    response = await ask(remote)
    assert (response.status_code, response.json()["error"]["code"]) == (502, "upstream_error")
    assert "zdr-detail" not in response.text


# ------------------------------------------------- pre controls run before egress


@pytest.mark.control("pii", "redact")
async def test_pii_is_redacted_before_the_prompt_leaves(remote, answers):
    content = f"Klient o numerze PESEL {PESEL} pyta o fakturę."
    response = await ask(remote, messages=[{"role": "user", "content": content}])
    assert response.status_code == 200
    leaving = sent(answers)["messages"][0]["content"]
    assert PESEL not in leaving
    assert "[REDACTED:" in leaving


@pytest.mark.control("secrets", "deny")
async def test_a_secret_never_leaves(remote, answers):
    response = await ask(remote, messages=[{"role": "user", "content": f"use {API_KEY}"}])
    assert response.status_code == 403
    assert not answers.called


@pytest.mark.control("prompt_injection", "deny")
async def test_a_confirmed_injection_never_leaves(remote, answers):
    remote.container.judges.answers["InjectionJudgement"] = {
        "is_injection": True,
        "confidence": 0.95,
        "rationale": "fake judge",
    }
    response = await ask(remote, messages=[{"role": "user", "content": f"Hi {INJECT_MARKER}"}])
    assert (response.status_code, response.json()["error"]["code"]) == (
        403,
        "prompt_injection_detected",
    )
    assert not answers.called


# ------------------------------------------------------------------------- judges


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool


async def test_judges_use_the_selected_upstream_with_mapping_terms_and_translation(
    tmp_path, openrouter
):
    route = openrouter.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=completion('{"ok": true}', model=PROVIDER))
    )
    async with running_gateway(
        tmp_path, llm_upstream="remote", env={KEY_ENV: KEY}, judge_factory=JudgeClient
    ) as gateway:
        verdict = await gateway.container.judges.judge(
            control_id="intent_judge",
            instructions="Is this fine?",
            content="hello",
            response_model=Verdict,
        )
    assert verdict.ok is True
    body = sent(route)
    assert body["model"] == PROVIDER  # llm_remote.judge_model, not judges.model (qwen3:8b)
    assert body["reasoning"] == {"enabled": False}  # judges.reasoning_effort: "none"
    assert body["provider"] == ZDR_TERMS["provider"]
    assert body["response_format"] == {"type": "json_object"}
    assert route.calls.last.request.headers["authorization"] == f"Bearer {KEY}"


# ------------------------------------------------------------------- visibility


async def test_audit_healthz_and_metric_name_the_remote_upstream(remote, answers):
    await ask(remote)
    (entry,) = remote.audit_entries()
    assert entry["upstream"] == "remote"
    health = (await remote.operator.get("/healthz")).json()
    assert (health["llm_upstream"], health["llm_upstream_host"]) == ("remote", "openrouter.ai")
    assert KEY not in json.dumps(health)
    assert REGISTRY.get_sample_value("acl_llm_upstream_info", {"upstream": "remote"}) == 1
    assert REGISTRY.get_sample_value("acl_llm_upstream_info", {"upstream": "local"}) == 0


async def test_the_local_default_is_unchanged(gateway, llm_upstream):
    route = llm_upstream.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=completion())
    )
    response = await ask(gateway, model="qwen3:8b", reasoning_effort="none", seed=7)
    assert response.status_code == 200
    body = sent(route)
    assert (body["model"], body["reasoning_effort"], body["seed"]) == ("qwen3:8b", "none", 7)
    assert "provider" not in body
    assert "reasoning" not in body
    assert "authorization" not in route.calls.last.request.headers
    (entry,) = gateway.audit_entries()
    assert entry["upstream"] == "local"
    health = (await gateway.operator.get("/healthz")).json()
    assert health["llm_upstream"] == "local"


# ------------------------------------------------------------------ startup refusal


def test_startup_refuses_remote_without_its_key(tmp_path):
    settings = make_settings(ROOT_POLICY, llm_upstream="remote", pins_dir=tmp_path)
    with pytest.raises(StartupError, match=KEY_ENV):
        GatewayContainer.from_settings(settings, env={})
    with pytest.raises(StartupError, match=KEY_ENV):
        GatewayContainer.from_settings(settings, env={KEY_ENV: "  "})


def test_startup_refuses_remote_without_the_policy_section(tmp_path):
    document = yaml.safe_load(ROOT_POLICY.read_text())
    del document["upstreams"]["llm_remote"]
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(document))
    settings = make_settings(path, llm_upstream="remote", pins_dir=tmp_path)
    with pytest.raises(PolicyLoadError, match="llm_remote"):
        GatewayContainer.from_settings(settings, env={KEY_ENV: KEY})


def test_remote_judges_need_a_mapped_judge_model(tmp_path):
    document = yaml.safe_load(ROOT_POLICY.read_text())
    del document["upstreams"]["llm_remote"]["judge_model"]
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(document))
    settings = make_settings(path, llm_upstream="remote", pins_dir=tmp_path)
    with pytest.raises(PolicyLoadError, match="judge_model"):
        GatewayContainer.from_settings(settings, env={KEY_ENV: KEY})
    # Locally the judges keep judges.model: the remote judge_model is not needed.
    PolicyLoader(requirement=require_upstream(LlmUpstreamKind.LOCAL)).load(path)


async def test_a_reload_cannot_drop_the_selected_remote_upstream(remote, answers):
    document = yaml.safe_load(remote.policy_path.read_text())
    del document["upstreams"]["llm_remote"]
    remote.policy_path.write_text(yaml.safe_dump(document))
    assert remote.container.policy_store.reload().result is ReloadResult.INVALID
    assert (await ask(remote)).status_code == 200  # still remote, never silently local


def test_local_mode_does_not_require_the_remote_section(tmp_path):
    document = yaml.safe_load(ROOT_POLICY.read_text())
    del document["upstreams"]["llm_remote"]
    PolicyLoader(requirement=require_upstream(LlmUpstreamKind.LOCAL)).parse(
        yaml.safe_dump(document).encode()
    )


# ------------------------------------------------------------------- schema rules


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"base_url": "http://openrouter.ai/api/v1"}, "https"),
        ({"extra_body": {"model": "openai/gpt-4o"}}, "gateway-owned"),
        ({"model_map": {LOGICAL: PROVIDER, "other:1b": PROVIDER}}, "one logical id"),
        ({"judge_model": "qwen3:8b"}, "judge_model 'qwen3:8b' has no model_map entry"),
        ({"pricing": {"llama3:70b": {"prompt_per_1k": 0.1}}}, "without a model_map entry"),
        ({"model_map": {}}, "at least 1"),
    ],
)
def test_the_remote_section_is_validated(change, message):
    document = yaml.safe_load(ROOT_POLICY.read_text())
    document["upstreams"]["llm_remote"].update(change)
    with pytest.raises(PolicyLoadError, match=message):
        PolicyLoader().parse(yaml.safe_dump(document).encode())


def test_outgoing_request_for_the_remote_upstream(snapshot):
    remote = snapshot.policy.upstreams.llm_remote
    assert remote is not None
    body = outgoing_request(
        {"model": LOGICAL, "messages": [], "stream": True, "stream_options": {}}, remote
    )
    assert body == {"model": PROVIDER, "messages": [], "stream": False, **ZDR_TERMS}


# ------------------------------------------------------------------------ budgets


async def test_remote_calls_charge_remote_token_prices_and_no_gpu_time(snapshot):
    clock = MutableClock(T0)
    ledger = BudgetLedger(
        InMemoryBudgetStore(clock=clock), clock=clock, llm_upstream=LlmUpstreamKind.REMOTE
    )
    call = BudgetedCall(
        session_id="s-1",
        principal="anna@demo",
        agent="databot",
        channel=Channel.LLM,
        model=LOGICAL,
        payload=chat(LOGICAL, max_tokens=100),
        user_label="anna@demo",
        agent_label="databot",
    )
    reservation = await ledger.reserve(call, snapshot)
    assert reservation.held.gpu_ms == 0
    assert reservation.gpu_allowance_s == snapshot.policy.limits.upstream_timeout_s  # deadline
    assert reservation.gpu_metered is False
    usage = TokenUsage(model=LOGICAL, prompt_tokens=1000, completion_tokens=1000)
    await ledger.settle(reservation, UpstreamResult(body={}, elapsed_s=42.0, usage=usage))
    await ledger.drain()
    spent = await ledger.store.usages(reservation.scopes)
    by_kind = {
        scope.kind.value: used for scope, used in zip(reservation.scopes, spent, strict=True)
    }
    # 1000 prompt * $0.0003/1k + 1000 completion * $0.0012/1k = $0.0015 (the max_price
    # ceiling, never less than an endpoint can charge); no GPU seconds.
    assert by_kind["per_user"].cost_nano_usd == 1_500_000
    assert all(used.gpu_ms == 0 for used in spent)  # 42 s of wall time, none of it ours
    # per_session limits GPU seconds and tool calls only: a remote LLM call draws on neither,
    # so the session scope (gpu_seconds: 900) is not even touched.
    assert "per_session" not in by_kind
    assert all(Meter.GPU not in scope.limits.limited() for scope in reservation.scopes)


# ------------------------------------------- settling on the cost OpenRouter reports

# completion()'s usage: 12 prompt + 7 completion tokens. At the max_price ceiling
# ($0.0003 / $0.0012 per 1k) that is 3,600 + 8,400 = 12,000 nano-USD: the most it may cost.
CEILING_USD = 0.000012


def cost_metric() -> float:
    labels = {"user": "anna@demo", "agent": "databot", "model": LOGICAL}
    return REGISTRY.get_sample_value("acl_cost_usd_total", labels) or 0.0


def billed(openrouter: respx.MockRouter, cost: object) -> None:
    """OpenRouter's answer with ``usage.cost`` set to ``cost`` (raw JSON text, so NaN works)."""
    body = completion(model=PROVIDER)
    body["usage"]["cost"] = "__COST__"
    raw = json.dumps(body).replace(
        '"__COST__"', cost if isinstance(cost, str) else json.dumps(cost)
    )
    openrouter.post("/chat/completions").mock(
        return_value=httpx.Response(
            200, content=raw.encode(), headers={"content-type": "application/json"}
        )
    )


async def settled_cost(remote: Harness) -> float:
    before = cost_metric()
    response = await ask(remote)
    assert response.status_code == 200
    await remote.container.budgets.drain()
    return round(cost_metric() - before, 12)


async def test_a_reported_cost_is_what_gets_settled(remote, openrouter):
    billed(openrouter, 0.000002)  # an endpoint well below the ceiling
    assert await settled_cost(remote) == pytest.approx(0.000002, abs=1e-12)


async def test_without_a_reported_cost_the_ceiling_is_settled(remote, openrouter):
    openrouter.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=completion(model=PROVIDER))
    )
    assert await settled_cost(remote) == pytest.approx(CEILING_USD, abs=1e-12)


async def test_a_reported_cost_above_the_ceiling_is_capped_and_logged(remote, openrouter, caplog):
    billed(openrouter, 0.5)
    with caplog.at_level(logging.WARNING, logger="gateway.budget.metering"):
        assert await settled_cost(remote) == pytest.approx(CEILING_USD, abs=1e-12)
    assert any("above_price_ceiling" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("cost", ["-0.001", "NaN", "Infinity", '"0.000002"', "true", "null"])
async def test_a_malformed_reported_cost_settles_at_the_ceiling(remote, openrouter, cost):
    billed(openrouter, cost)
    assert await settled_cost(remote) == pytest.approx(CEILING_USD, abs=1e-12)
