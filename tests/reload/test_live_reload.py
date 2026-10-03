"""Live reload through the operator API: a policy edit changes the verdict without a restart."""

import httpx
import pytest
from gateway_testkit import bearer, chat, completion

CHAT = "/v1/chat/completions"
GRANT_LLAMA = (
    '"generate:model:qwen3:8b"]',
    '"generate:model:qwen3:8b", "generate:model:llama3:70b"]',
)


@pytest.fixture
def upstream(llm_upstream):
    return llm_upstream.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=completion(model="llama3:70b"))
    )


def edit(gateway, old: str, new: str) -> None:
    text = gateway.policy_path.read_text()
    assert old in text
    gateway.policy_path.write_text(text.replace(old, new, 1))


async def reload(gateway, sub: str = "root@demo") -> httpx.Response:
    token = await gateway.token(sub)
    return await gateway.operator.post("/admin/reload", headers=bearer(token))


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


@pytest.mark.parametrize("sub", ["anna@demo", "olga@demo", "svc:nightly_etl"])
async def test_reload_needs_the_admin_role(gateway, sub):
    edit(gateway, *GRANT_LLAMA)
    revision = gateway.container.policy_store.current.revision
    response = await reload(gateway, sub)
    assert (response.status_code, response.json()["error"]["code"]) == (403, "admin_required")
    assert gateway.container.policy_store.current.revision == revision


async def test_reload_needs_a_token(gateway):
    response = await gateway.operator.post("/admin/reload")
    assert (response.status_code, response.json()["error"]["code"]) == (401, "token_missing")
