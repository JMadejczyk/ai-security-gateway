"""The approval queue end to end (SPEC "Human in the loop", demo step 3), through both apps
and real in-process MCP servers.

The autonomous ``nightly_etl`` reads a page with a hidden injection (taint), so its report
write is held for approval instead of refused. An approver (``olga@demo``, ops-team) sees
it at ``/admin/approvals`` and approves; the agent retries the exact call with the approval
id in ``_meta`` and it executes once.
"""

import importlib
from typing import Any

import httpx
import jwt
import pytest
import yaml
from approvals_kit import mcp_harness
from gateway_testkit import bearer, chat, completion

from gateway.approvals.model import ApprovalState
from gateway.controls.registry import ControlRegistry
from gateway.core.envelope import Verdict
from gateway.core.interfaces import Control
from gateway.core.types import ControlKind, Decision, Stage
from gateway.sessions import SessionUpdate
from gateway.telemetry import REGISTRY, ReloadResult

connect_all = mcp_harness.connect_all
error_text = mcp_harness.error_text

ETL, OLGA, ROOT, BARTEK, ANNA = (
    "svc:nightly_etl",
    "olga@demo",
    "root@demo",
    "bartek@demo",
    "anna@demo",
)
META = "ai-control-layer/approval_id"
REPORT = {"name": "nightly.md", "content": "numbers"}
ORDERS = "SELECT COUNT(*) FROM sales.orders"


async def tainted_etl(stack: Any) -> Any:
    """The ETL's reports client, in a session the fetch tool has tainted."""
    web, reports = await connect_all(stack, ETL, "web", "reports")
    await web.call("fetch", url="https://example.com/outlook")
    return reports


async def held(reports: Any, **arguments: Any) -> str:
    result = await reports.call("write_report", **(REPORT | arguments))
    assert error_text(result).startswith("approval_required"), result
    return result["_meta"][META]


async def retry(reports: Any, approval_id: object, **arguments: Any) -> dict[str, Any]:
    params = {
        "name": "write_report",
        "arguments": REPORT | arguments,
        "_meta": {META: approval_id},
    }
    response = await reports.request("tools/call", params)
    assert response.status_code == 200, response.text
    return response.json()["result"]


async def operator(stack: Any, sub: str) -> dict[str, str]:
    return bearer(await stack.gateway.token(sub))


async def decide(
    stack: Any, sub: str, approval_id: str, verb: str = "approve", **body: Any
) -> httpx.Response:
    return await stack.gateway.operator.post(
        f"/admin/approvals/{approval_id}/{verb}",
        json=body or None,
        headers=await operator(stack, sub),
    )


async def state_of(stack: Any, approval_id: str) -> ApprovalState:
    record = await stack.gateway.container.oversight.approvals.get(approval_id)
    assert record is not None
    return record.state


def approvals_total(decision: str) -> float:
    return REGISTRY.get_sample_value("acl_approvals_total", {"decision": decision}) or 0.0


# ------------------------------------------------------------------- demo step 3


async def test_held_write_is_approved_and_runs_exactly_once(stack: Any):
    reports = await tainted_etl(stack)
    approved_before = approvals_total("approved")
    approval_id = await held(reports)
    assert stack.log.of("write_report") == []
    assert REGISTRY.get_sample_value("acl_approvals_pending", {}) == 1

    listing = await stack.gateway.operator.get(
        "/admin/approvals", params={"state": "pending"}, headers=await operator(stack, OLGA)
    )
    assert listing.status_code == 200, listing.text
    (entry,) = listing.json()["approvals"]
    assert entry["id"] == approval_id
    assert (entry["agent"], entry["principal"], entry["tool"]) == (ETL[4:], ETL, "write_report")
    assert entry["resources"] == ["fs:reports/nightly.md"]
    assert entry["reasons"] == ["session_requires_approval"]
    assert entry["policy_revision"] == stack.gateway.container.policy_store.current.revision
    assert "numbers" not in listing.text  # never the arguments

    approved = await decide(stack, OLGA, approval_id, note="nightly run")
    assert approved.status_code == 200, approved.text
    assert (approved.json()["state"], approved.json()["decided_by"]) == ("approved", OLGA)
    assert approvals_total("approved") == approved_before + 1
    assert REGISTRY.get_sample_value("acl_approvals_pending", {}) == 0

    done = await retry(reports, approval_id)
    assert done["isError"] is False, done
    assert len(stack.log.of("write_report")) == 1
    assert await state_of(stack, approval_id) is ApprovalState.SUCCEEDED
    entry = stack.gateway.audit_entries()[-1]
    assert (entry["decision"], entry["approval_id"]) == ("allow", approval_id)

    # The approval is spent: replaying it never runs the write again.
    again = await retry(reports, approval_id)
    assert error_text(again) == "approval_already_used"
    assert len(stack.log.of("write_report")) == 1


async def test_altered_arguments_are_a_mismatch_and_do_not_consume(stack: Any):
    reports = await tainted_etl(stack)
    approval_id = await held(reports)
    assert (await decide(stack, OLGA, approval_id)).status_code == 200

    altered = await retry(reports, approval_id, content="numbers; and the payroll")
    assert error_text(altered) == "approval_mismatch"
    other_tool = await reports.request(
        "tools/call",
        {"name": "drop_reports", "arguments": {}, "_meta": {META: approval_id}},
    )
    assert error_text(other_tool.json()["result"]) in {"approval_mismatch", "tool_not_mapped"}
    assert stack.log.of("write_report") == []
    assert await state_of(stack, approval_id) is ApprovalState.APPROVED

    assert (await retry(reports, approval_id))["isError"] is False  # the exact call still runs


async def test_another_session_cannot_use_the_approval(stack: Any):
    reports = await tainted_etl(stack)
    approval_id = await held(reports)
    assert (await decide(stack, OLGA, approval_id)).status_code == 200
    intruder = await tainted_etl(stack)  # same agent and principal, another session
    assert error_text(await retry(intruder, approval_id)) == "approval_mismatch"
    assert await state_of(stack, approval_id) is ApprovalState.APPROVED


@pytest.mark.parametrize("bogus", ["apr-" + "0" * 24, "not-an-id", "", 42])
async def test_unknown_approval_ids_are_refused(stack: Any, bogus: object):
    reports = await tainted_etl(stack)
    assert error_text(await retry(reports, bogus)) == "approval_unknown"
    assert stack.log.of("write_report") == []


async def test_retrying_a_pending_operation_returns_the_same_id(stack: Any):
    reports = await tainted_etl(stack)
    approval_id = await held(reports)
    assert await held(reports) == approval_id  # without the id
    still = await retry(reports, approval_id)  # with it
    assert still["_meta"][META] == approval_id
    records = await stack.gateway.container.oversight.approvals.store.records()
    assert [r.id for r in records] == [approval_id]


async def test_approval_retries_do_not_trip_loop_detection(stack: Any):
    reports = await tainted_etl(stack)
    approval_id = await held(reports)
    for _ in range(8):  # max_repeats is 5 in 60 s
        assert (await retry(reports, approval_id))["_meta"][META] == approval_id
    assert (await decide(stack, OLGA, approval_id)).status_code == 200
    assert (await retry(reports, approval_id))["isError"] is False

    # The same repeats without an approval id are a loop.
    reasons = [
        error_text(await reports.call("write_report", name="other.md", content="x"))
        for _ in range(6)
    ]
    assert reasons[-1] == "loop_detected"


# ------------------------------------------------------------------- who decides


async def test_an_agent_without_an_approver_role_cannot_see_or_decide(stack: Any):
    reports = await tainted_etl(stack)
    approval_id = await held(reports)
    headers = await operator(stack, BARTEK)
    listing = await stack.gateway.operator.get("/admin/approvals", headers=headers)
    assert listing.status_code == 403
    assert listing.json()["error"]["code"] == "operator_role_required"
    refused = await decide(stack, BARTEK, approval_id)
    assert (refused.status_code, refused.json()["error"]["code"]) == (
        403,
        "operator_role_required",
    )
    no_token = await stack.gateway.operator.get("/admin/approvals")
    assert no_token.status_code == 401
    assert await state_of(stack, approval_id) is ApprovalState.PENDING


async def test_admin_can_approve(stack: Any):
    reports = await tainted_etl(stack)
    approval_id = await held(reports)
    approved = await decide(stack, ROOT, approval_id)
    assert (approved.status_code, approved.json()["decided_by"]) == (200, ROOT)
    assert (await retry(reports, approval_id))["isError"] is False


async def test_self_approval_is_refused(stack: Any):
    """olga's operator token acts through databot: she may not decide databot's approvals."""
    (reports,) = await connect_all(stack, ANNA, "reports")
    await stack.gateway.container.sessions.apply(
        _session_of(reports.token), SessionUpdate(risk_delta=0.6), half_life_s=600
    )
    approval_id = await held(reports)  # interactive risk > 0.5: write needs approval
    refused = await decide(stack, OLGA, approval_id)
    assert (refused.status_code, refused.json()["error"]["code"]) == (
        403,
        "self_approval_forbidden",
    )
    assert await state_of(stack, approval_id) is ApprovalState.PENDING


async def test_a_decided_approval_cannot_be_decided_again(stack: Any):
    reports = await tainted_etl(stack)
    approval_id = await held(reports)
    assert (await decide(stack, OLGA, approval_id)).status_code == 200
    again = await decide(stack, ROOT, approval_id)
    assert (again.status_code, again.json()["error"]["code"]) == (409, "approval_not_pending")
    missing = await decide(stack, OLGA, "apr-" + "1" * 24)
    assert missing.status_code == 404


# ------------------------------------------------------------- denial and expiry


async def test_denied_approval_blocks_the_retry(stack: Any):
    reports = await tainted_etl(stack)
    approval_id = await held(reports)
    denied = await decide(stack, OLGA, approval_id, "deny", note="not tonight")
    assert (denied.json()["state"], denied.json()["note"]) == ("denied", "not tonight")
    assert error_text(await retry(reports, approval_id)) == "approval_denied"
    assert stack.log.of("write_report") == []


async def test_an_unused_approval_can_be_revoked(stack: Any):
    reports = await tainted_etl(stack)
    approval_id = await held(reports)
    assert (await decide(stack, OLGA, approval_id)).status_code == 200
    assert (await decide(stack, OLGA, approval_id, "deny")).status_code == 200
    assert error_text(await retry(reports, approval_id)) == "approval_denied"


async def test_timeout_expires_the_approval(stack: Any):
    reports = await tainted_etl(stack)
    approval_id = await held(reports)
    stack.gateway.clock.advance(601)  # approvals.timeout_s: 600, on_timeout: deny
    assert await stack.gateway.container.oversight.approvals.expire_due() == 1
    assert REGISTRY.get_sample_value("acl_approvals_pending", {}) == 0
    too_late = await decide(stack, OLGA, approval_id)
    assert (too_late.status_code, too_late.json()["error"]["message"]) == (
        409,
        "the approval is expired",
    )
    assert error_text(await retry(reports, approval_id)) == "approval_expired"
    assert stack.log.of("write_report") == []


async def test_an_approval_not_used_in_time_expires(stack: Any):
    reports = await tainted_etl(stack)
    approval_id = await held(reports)
    assert (await decide(stack, OLGA, approval_id)).status_code == 200
    stack.gateway.clock.advance(601)
    assert error_text(await retry(reports, approval_id)) == "approval_expired"
    assert stack.log.of("write_report") == []


# ------------------------------------------------------------------- hot reload


async def test_a_reload_revoking_the_grant_stops_an_approved_call(stack: Any):
    reports = await tainted_etl(stack)
    approval_id = await held(reports)
    assert (await decide(stack, OLGA, approval_id)).status_code == 200

    document = yaml.safe_load(stack.gateway.policy_path.read_text())
    agent = document["agents"]["nightly_etl"]
    agent["allow"] = [g for g in agent["allow"] if not g.startswith("write:")]
    stack.gateway.policy_path.write_text(yaml.safe_dump(document))
    assert stack.gateway.container.policy_store.reload().result is ReloadResult.OK

    refused = await retry(reports, approval_id)
    assert error_text(refused) == "outside_agent_scope"  # an approval grants nothing
    assert stack.log.of("write_report") == []
    assert await state_of(stack, approval_id) is ApprovalState.APPROVED  # not consumed


async def test_an_approval_survives_an_unrelated_reload(stack: Any):
    reports = await tainted_etl(stack)
    approval_id = await held(reports)
    assert (await decide(stack, OLGA, approval_id)).status_code == 200
    document = yaml.safe_load(stack.gateway.policy_path.read_text())
    document["approvals"]["timeout_s"] = 900
    stack.gateway.policy_path.write_text(yaml.safe_dump(document))
    assert stack.gateway.container.policy_store.reload().result is ReloadResult.OK
    assert (await retry(reports, approval_id))["isError"] is False  # re-checked, still allowed


# ------------------------------------------------------------- upstream outcomes


async def test_upstream_timeout_after_send_is_uncertain_and_never_retried(
    stack: Any, monkeypatch: pytest.MonkeyPatch
):
    reports = await tainted_etl(stack)
    approval_id = await held(reports)
    assert (await decide(stack, OLGA, approval_id)).status_code == 200

    original = stack.transport.handle_async_request
    attempts: list[httpx.Request] = []

    async def lose_the_answer(request: httpx.Request) -> httpx.Response:
        if request.url.host == "mcp-files" and b'"write_report"' in request.content:
            attempts.append(request)
            raise httpx.ReadTimeout("answer lost", request=request)
        return await original(request)

    monkeypatch.setattr(stack.transport, "handle_async_request", lose_the_answer)
    lost = await retry(reports, approval_id)
    assert error_text(lost) == "upstream_timeout"
    record = await stack.gateway.container.oversight.approvals.get(approval_id)
    assert (record.state, record.outcome) == (ApprovalState.UNCERTAIN, "upstream_timeout")

    replay = await retry(reports, approval_id)
    assert error_text(replay) == "approval_outcome_uncertain"
    assert len(attempts) == 1  # the gateway never sent it again


async def test_an_upstream_error_answer_is_a_failed_approval(stack: Any):
    reports = await tainted_etl(stack)
    approval_id = await held(reports)
    assert (await decide(stack, OLGA, approval_id)).status_code == 200
    stack.transport.fail_tool_calls.add("mcp-files")
    assert error_text(await retry(reports, approval_id)) == "upstream_error"
    assert await state_of(stack, approval_id) is ApprovalState.FAILED


# ------------------------------------------------------------------- LLM channel


class HoldEverything(Control):
    """A scripted ``pii`` that wants a human to look at every prompt."""

    id = "pii"
    stages = frozenset({Stage.PRE})
    kind = ControlKind.DETERMINISTIC

    async def evaluate(self, interaction, stage, cfg):
        return Verdict(
            decision=Decision.REQUIRE_APPROVAL, control_id=self.id, reason_code="needs_review"
        )


async def test_llm_hold_answers_403_with_the_id_and_a_header_retry_runs_once(gateway, llm_upstream):
    upstream = llm_upstream.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=completion())
    )
    controls: ControlRegistry = gateway.container.pipeline.controls
    controls.clear()
    controls.register(HoldEverything())
    token = await gateway.token(ETL)

    first = await gateway.agent.post("/v1/chat/completions", json=chat(), headers=bearer(token))
    assert first.status_code == 403
    error = first.json()["error"]
    assert error["code"] == "approval_required"
    approval_id = error["approval_id"]
    assert not upstream.called

    olga = bearer(await gateway.token(OLGA))
    view = await gateway.operator.get(f"/admin/approvals/{approval_id}", headers=olga)
    assert (view.json()["channel"], view.json()["reasons"]) == ("llm", ["needs_review"])
    approve = await gateway.operator.post(f"/admin/approvals/{approval_id}/approve", headers=olga)
    assert approve.status_code == 200

    headers = bearer(token) | {"x-acl-approval-id": approval_id}
    done = await gateway.agent.post("/v1/chat/completions", json=chat(), headers=headers)
    assert done.status_code == 200, done.text
    assert upstream.call_count == 1
    replay = await gateway.agent.post("/v1/chat/completions", json=chat(), headers=headers)
    assert (replay.status_code, replay.json()["error"]["code"]) == (403, "approval_already_used")
    assert upstream.call_count == 1


def _session_of(token: str) -> str:
    return jwt.decode(token, options={"verify_signature": False})[
        "session_id"
    ]  # verified by the gateway


# ------------------------------------------------------------- intent_judge flags


async def test_an_intent_flagged_mcp_call_goes_through_the_same_queue(stack: Any):
    """The LLM answer is released; the flagged tool_call's MCP call is held (intent_flagged),
    approved, and the retry with the id runs it once."""
    judges = importlib.import_module("test_judges")  # tests/mcp: the scripted judge upstream
    llm = judges.ScriptedLLM()
    judges.llm_route(stack, llm)
    judges.enable_judges(stack.gateway, intent_judge={"risk_delta": 0.2})
    llm.aligned = lambda _call: False
    llm.agent_answer = completion(None, tool_calls=[judges.tool_call("query", {"sql": ORDERS})])
    (db,) = await connect_all(stack, ETL, "sales_db")
    assert (await judges.ask(stack.gateway, db.token)).status_code == 200

    first = await db.call("query", sql=ORDERS)
    assert error_text(first).startswith("approval_required"), first
    approval_id = first["_meta"][META]
    olga = await operator(stack, OLGA)
    view = await stack.gateway.operator.get(f"/admin/approvals/{approval_id}", headers=olga)
    assert "intent_flagged" in view.json()["reasons"]
    assert (await decide(stack, OLGA, approval_id)).status_code == 200

    params = {"name": "query", "arguments": {"sql": ORDERS}, "_meta": {META: approval_id}}
    done = (await db.request("tools/call", params)).json()["result"]
    assert done["isError"] is False, done
    assert len(stack.log.of("query")) == 1
    assert await state_of(stack, approval_id) is ApprovalState.SUCCEEDED
