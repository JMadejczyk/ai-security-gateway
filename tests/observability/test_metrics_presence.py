"""Every metric SPEC "Audit, metrics and Grafana" lists is exported with its labels.

The gateway is exercised in process (LLM calls through the agent app against a mocked
upstream, a policy reload, an autonomous session pushed over its throttle), then
``/metrics`` on the operator app is parsed. Label sets are checked exactly: no session id,
principal or free text ever becomes a label.
"""

from typing import Any

import jwt
import pytest
import respx
import yaml
from gateway_testkit import Harness, bearer, chat, echo_completion
from prometheus_client.parser import text_string_to_metric_families

from gateway.sessions import SessionUpdate

CHAT = "/v1/chat/completions"

# SPEC metric -> (sample name, label names without `le`)
SPEC_METRICS: dict[str, tuple[str, frozenset[str]]] = {
    "acl_requests_total": ("acl_requests_total", frozenset({"channel", "decision", "agent"})),
    "acl_control_verdicts_total": (
        "acl_control_verdicts_total",
        frozenset({"control", "decision"}),
    ),
    "acl_control_latency_seconds": ("acl_control_latency_seconds_bucket", frozenset({"control"})),
    "acl_overhead_seconds": ("acl_overhead_seconds_bucket", frozenset({"channel"})),
    "acl_tokens_total": ("acl_tokens_total", frozenset({"user", "agent", "model"})),
    "acl_cost_usd_total": ("acl_cost_usd_total", frozenset({"user", "agent", "model"})),
    "acl_budget_usage_ratio": ("acl_budget_usage_ratio", frozenset({"scope", "id"})),
    "acl_session_risk": ("acl_session_risk_bucket", frozenset()),
    "acl_tainted_sessions": ("acl_tainted_sessions", frozenset()),
    "acl_approvals_pending": ("acl_approvals_pending", frozenset()),
    "acl_throttled_total": ("acl_throttled_total", frozenset({"agent"})),
    "acl_policy_reloads_total": ("acl_policy_reloads_total", frozenset({"result"})),
    "acl_policy_info": ("acl_policy_info", frozenset({"revision"})),
}
KNOWN_AGENTS = {"databot", "nightly_etl", "other"}
UNPRICED_MODEL = "llama3.1:8b"  # granted by analyst/databot's generate:model:*, not in pricing


async def _exercise(gateway: Harness, router: respx.MockRouter) -> None:
    router.post("/chat/completions").mock(side_effect=echo_completion)
    # Token prices, so the call has a cost (the root policy prices qwen3:8b by GPU time only).
    document: dict[str, Any] = yaml.safe_load(gateway.policy_path.read_text())
    document["pricing"]["qwen3:8b"].update(prompt_per_1k=0.01, completion_per_1k=0.02)
    gateway.policy_path.write_text(yaml.safe_dump(document))
    reload = await gateway.operator.post(
        "/admin/reload", headers=bearer(await gateway.operator_token("root@demo"))
    )
    assert reload.json()["result"] == "ok", reload.text

    anna = await gateway.token("anna@demo")
    assert (await gateway.agent.post(CHAT, json=chat(), headers=bearer(anna))).status_code == 200
    # analyst/databot may generate with any model; one without a pricing entry is `other`.
    unpriced = await gateway.agent.post(CHAT, json=chat(UNPRICED_MODEL), headers=bearer(anna))
    assert unpriced.status_code == 200, unpriced.text

    # An autonomous session above risk 0.5 is throttled to one call per 10 s.
    etl = await gateway.token("svc:nightly_etl")
    assert (await gateway.agent.post(CHAT, json=chat(), headers=bearer(etl))).status_code == 200
    session_id = jwt.decode(etl, options={"verify_signature": False})["session_id"]
    await gateway.container.sessions.apply(
        session_id, SessionUpdate(risk_delta=0.7), half_life_s=600
    )
    statuses = [
        (await gateway.agent.post(CHAT, json=chat(), headers=bearer(etl))).status_code
        for _ in range(2)
    ]
    assert statuses == [200, 429]


@pytest.fixture
async def exposition(gateway: Harness, llm_upstream: respx.MockRouter) -> dict[str, list[Any]]:
    await _exercise(gateway, llm_upstream)
    response = await gateway.operator.get("/metrics")
    assert response.status_code == 200
    samples: dict[str, list[Any]] = {}
    for family in text_string_to_metric_families(response.text):
        for sample in family.samples:
            samples.setdefault(sample.name, []).append(sample)
    return samples


@pytest.mark.parametrize("metric", sorted(SPEC_METRICS))
async def test_spec_metric_is_exported_with_its_labels(
    exposition: dict[str, list[Any]], metric: str
) -> None:
    sample_name, labels = SPEC_METRICS[metric]
    samples = exposition.get(sample_name, [])
    assert samples, f"{metric} missing from /metrics"
    for sample in samples:
        assert set(sample.labels) - {"le"} == labels, (metric, sample.labels)


async def test_exercised_metrics_carry_values(exposition: dict[str, list[Any]]) -> None:
    def total(name: str, **labels: str) -> float:
        return sum(
            s.value
            for s in exposition.get(name, [])
            if all(s.labels.get(k) == v for k, v in labels.items())
        )

    assert total("acl_requests_total", channel="llm", decision="allow", agent="databot") >= 1
    assert total("acl_tokens_total", user="anna@demo", agent="databot", model="qwen3:8b") >= 19
    assert total("acl_cost_usd_total", user="anna@demo", agent="databot") > 0
    assert total("acl_throttled_total", agent="nightly_etl") >= 1
    assert total("acl_policy_reloads_total", result="ok") >= 1
    assert total("acl_session_risk_count") >= 1
    assert total("acl_overhead_seconds_count", channel="llm") >= 1
    assert total("acl_budget_usage_ratio", scope="per_user.daily_tokens", id="anna@demo") > 0
    revisions = exposition["acl_policy_info"]
    assert len(revisions) == 1
    assert revisions[0].value == 1


async def test_models_outside_the_pricing_table_are_bucketed_as_other(
    exposition: dict[str, list[Any]],
) -> None:
    tokens = {s.labels["model"]: s.value for s in exposition["acl_tokens_total"]}
    assert tokens.get("other", 0) >= 19  # the unpriced call's usage
    assert tokens.get("qwen3:8b", 0) >= 19
    for name in ("acl_tokens_total", "acl_cost_usd_total"):
        assert {s.labels["model"] for s in exposition[name]} <= {"qwen3:8b", "other"}, name


async def test_labels_stay_bounded(exposition: dict[str, list[Any]]) -> None:
    for name in ("acl_requests_total", "acl_throttled_total"):
        assert {s.labels["agent"] for s in exposition[name]} <= KNOWN_AGENTS
    for samples in exposition.values():
        for sample in samples:
            assert not {"session_id", "principal", "resource", "reason_code"} & set(
                sample.labels
            ), sample


async def test_decision_counters_start_at_zero_for_every_label_set(
    exposition: dict[str, list[Any]],
) -> None:
    """Pre-created series: increase() on a dashboard sees each label set's first event."""
    requests = {
        (s.labels["channel"], s.labels["decision"], s.labels["agent"])
        for s in exposition["acl_requests_total"]
    }
    for channel in ("llm", "mcp"):
        for decision in ("allow", "redact", "block", "require_approval"):
            for agent in KNOWN_AGENTS:
                assert (channel, decision, agent) in requests
    verdicts = {
        (s.labels["control"], s.labels["decision"])
        for s in exposition["acl_control_verdicts_total"]
    }
    assert ("sql_guard", "block") in verdicts
    assert ("prompt_injection", "block") in verdicts
    signatures = {s.labels["signature"] for s in exposition["acl_signature_hits_total"]}
    assert "inj.ignore-previous" in signatures
    approvals = {s.labels["decision"] for s in exposition["acl_approvals_total"]}
    assert {"pending", "approved", "denied", "expired"} <= approvals
