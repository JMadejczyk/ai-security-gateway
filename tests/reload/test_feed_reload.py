"""Demo step 7: a signature added to the feed changes the verdict without a restart, and the
audit entry shows the new feed version."""

import json

import pytest
from gateway_testkit import bearer, chat, echo_completion

from gateway.feed.schema import parse_feed
from gateway.telemetry import FeedReloadResult

ALLOW = pytest.mark.control("signatures", "allow")
DENY = pytest.mark.control("signatures", "deny")

CHAT = "/v1/chat/completions"
PROMPT = "Run the quarterly purge-sequence on the sales tables"


async def ask(gateway, token: str):
    body = chat(messages=[{"role": "user", "content": PROMPT}])
    return await gateway.agent.post(CHAT, json=body, headers=bearer(token))


def edit_feed(gateway, version: str, *, extra: list[dict] | None = None, raw: str | None = None):
    path = gateway.policy_path.parent / "feeds" / "signatures.json"
    if raw is not None:
        path.write_text(raw)
        return
    feed = json.loads(path.read_text())
    feed["version"] = version
    feed["signatures"].extend(extra or [])
    path.write_text(json.dumps(feed))


PURGE = {
    "id": "judge.purge-sequence",
    "source": "added live by a judge",
    "pattern_type": "regex",
    "pattern": "(?i)purge-sequence",
    "severity": "critical",
    "channels": ["llm"],
}


@ALLOW
@DENY
async def test_new_signature_changes_the_verdict_without_restart(gateway, llm_upstream):
    llm_upstream.post("/chat/completions").mock(side_effect=echo_completion)
    token = await gateway.token("anna@demo")
    store = gateway.container.feed_store
    old_version = store.version

    assert (await ask(gateway, token)).status_code == 200

    edit_feed(gateway, "judge.1", extra=[PURGE])
    outcome = await store.refresh()
    assert (outcome.result, outcome.version) == (FeedReloadResult.OK, "judge.1")

    after = await ask(gateway, token)
    assert (after.status_code, after.json()["error"]["code"]) == (403, "signature_match")
    before_entry, after_entry = gateway.audit_entries()
    assert (before_entry["feed_version"], after_entry["feed_version"]) == (old_version, "judge.1")
    assert before_entry["policy_revision"] == after_entry["policy_revision"]


@DENY
async def test_a_broken_feed_edit_keeps_the_last_valid_feed(gateway, llm_upstream):
    llm_upstream.post("/chat/completions").mock(side_effect=echo_completion)
    token = await gateway.token("anna@demo")
    store = gateway.container.feed_store
    edit_feed(gateway, "judge.1", extra=[PURGE])
    await store.refresh()

    edit_feed(gateway, "judge.2", raw='{"version": "judge.2", "signatures": [{"id": "x"}]}')
    outcome = await store.refresh()
    assert (outcome.result, outcome.version) == (FeedReloadResult.INVALID, "judge.1")

    response = await ask(gateway, token)
    assert response.json()["error"]["code"] == "signature_match"  # judge.1 still applies
    assert gateway.audit_entries()[-1]["feed_version"] == "judge.1"


async def test_the_gateway_refreshes_the_feed_in_the_background(gateway):
    assert gateway.container._feed_refresher is not None
    assert not gateway.container._feed_refresher.done()


@ALLOW
@DENY
async def test_a_refresh_mid_call_does_not_change_the_feed_judging_it(gateway, llm_upstream):
    """Pre and post use the feed pinned at admission, the one the audit entry names."""
    store = gateway.container.feed_store
    pinned = store.version
    answer_signature = {**PURGE, "id": "judge.answer", "pattern": "(?i)40 customers"}

    def answer_after_a_refresh(request):
        feed = json.loads((gateway.policy_path.parent / "feeds" / "signatures.json").read_text())
        feed["version"] = "judge.mid"
        feed["signatures"].append(answer_signature)
        store._swap(parse_feed(json.dumps(feed).encode()))  # a refresh landing mid-call
        return echo_completion(request)

    llm_upstream.post("/chat/completions").mock(side_effect=answer_after_a_refresh)
    token = await gateway.token("anna@demo")
    response = await gateway.agent.post(CHAT, json=chat(), headers=bearer(token))
    assert response.status_code == 200  # the post stage still used the pinned feed
    assert gateway.audit_entries()[-1]["feed_version"] == pinned
    blocked = await gateway.agent.post(CHAT, json=chat(), headers=bearer(token))
    assert blocked.json()["error"]["code"] == "signature_match"  # the next call sees judge.mid
    assert gateway.audit_entries()[-1]["feed_version"] == "judge.mid"
