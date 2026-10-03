"""Live reload through the operator API: a policy edit changes the verdict without a restart."""

import httpx
import pytest
from gateway_testkit import bearer, chat, echo_completion

ALLOW = pytest.mark.control("authz", "allow")
DENY = pytest.mark.control("authz", "deny")

CHAT = "/v1/chat/completions"
GRANT_LLAMA = (
    '"generate:model:qwen3:8b"]',
    '"generate:model:qwen3:8b", "generate:model:llama3:70b"]',
)


@pytest.fixture
def upstream(llm_upstream):
    return llm_upstream.post("/chat/completions").mock(side_effect=echo_completion)


def edit(gateway, old: str, new: str) -> None:
    text = gateway.policy_path.read_text()
    assert old in text
    gateway.policy_path.write_text(text.replace(old, new, 1))


async def reload(gateway, sub: str = "root@demo") -> httpx.Response:
    token = await gateway.operator_token(sub)
    return await gateway.operator.post("/admin/reload", headers=bearer(token))


@ALLOW
@DENY
async def test_reload_grants_the_model_without_restart(gateway, upstream):
    bartek = await gateway.token("bartek@demo")
    before = await gateway.agent.post(CHAT, json=chat("llama3:70b"), headers=bearer(bartek))
    assert before.status_code == 403
    old_revision = gateway.container.policy_store.current.revision

    edit(gateway, *GRANT_LLAMA)
    response = await reload(gateway)
    assert response.status_code == 200
    outcome = response.json()
    assert (outcome["result"], outcome["previous_revision"]) == ("ok", old_revision)
    assert outcome["revision"] != old_revision

    after = await gateway.agent.post(CHAT, json=chat("llama3:70b"), headers=bearer(bartek))
    assert after.status_code == 200
    denied, allowed = gateway.audit_entries()
    assert (denied["policy_revision"], allowed["policy_revision"]) == (
        old_revision,
        outcome["revision"],
    )


@ALLOW
@DENY
async def test_broken_file_keeps_the_previous_policy(gateway, upstream):
    bartek = await gateway.token("bartek@demo")
    edit(gateway, "default: deny", "default: allow")
    response = await reload(gateway)
    assert response.status_code == 422
    assert response.json()["result"] == "invalid"
    blocked = await gateway.agent.post(CHAT, json=chat("llama3:70b"), headers=bearer(bartek))
    assert blocked.status_code == 403
    allowed = await gateway.agent.post(CHAT, json=chat(), headers=bearer(bartek))
    assert allowed.status_code == 200


async def test_unchanged_file(gateway):
    response = await reload(gateway)
    assert (response.status_code, response.json()["result"]) == (200, "unchanged")


async def test_reload_needs_the_admin_role(gateway):
    edit(gateway, *GRANT_LLAMA)
    revision = gateway.container.policy_store.current.revision
    response = await reload(gateway, "olga@demo")  # an operator, but an approver only
    assert (response.status_code, response.json()["error"]["code"]) == (403, "admin_required")
    assert gateway.container.policy_store.current.revision == revision


@pytest.mark.control("authn", "deny")
@pytest.mark.parametrize("sub", ["root@demo", "anna@demo", "svc:nightly_etl"])
async def test_reload_refuses_agent_tokens(gateway, sub):
    edit(gateway, *GRANT_LLAMA)
    revision = gateway.container.policy_store.current.revision
    token = await gateway.token(sub)  # an agent token, even root's
    response = await gateway.operator.post("/admin/reload", headers=bearer(token))
    assert (response.status_code, response.json()["error"]["code"]) == (401, "wrong_audience")
    assert gateway.container.policy_store.current.revision == revision


async def test_reload_needs_a_token(gateway):
    response = await gateway.operator.post("/admin/reload")
    assert (response.status_code, response.json()["error"]["code"]) == (401, "token_missing")
