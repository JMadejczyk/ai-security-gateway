"""``python -m gateway.cli`` against the operator app over an in-process httpx transport."""

import json
from typing import Any

import httpx
import pytest
from approvals_kit import mcp_harness

from gateway.cli.__main__ import build_parser, run
from gateway.main import create_operator_app

connect_all = mcp_harness.connect_all

ETL, OLGA, ROOT, BARTEK = "svc:nightly_etl", "olga@demo", "root@demo", "bartek@demo"
META = "ai-control-layer/approval_id"
REPORT = {"name": "nightly.md", "content": "numbers"}


async def held_write(stack: Any) -> tuple[Any, str]:
    web, reports = await connect_all(stack, ETL, "web", "reports")
    await web.call("fetch", url="https://example.com/outlook")
    result = await reports.call("write_report", **REPORT)
    return reports, result["_meta"][META]


async def cli(stack: Any, sub: str | None, *argv: str) -> int:
    env = {"ACL_OPERATOR_TOKEN": await stack.gateway.token(sub)} if sub else {}
    transport = httpx.ASGITransport(app=create_operator_app(stack.gateway.container))
    return await run(list(argv), env=env, transport=transport)


async def test_list_show_approve_then_the_retry_runs(stack: Any, capsys: pytest.CaptureFixture):
    reports, approval_id = await held_write(stack)

    assert await cli(stack, OLGA, "approvals", "list") == 0
    table = capsys.readouterr().out.splitlines()
    assert table[0].split()[:3] == ["ID", "STATE", "AGENT"]
    assert table[1].split()[:3] == [approval_id, "pending", "nightly_etl"]
    assert "numbers" not in "\n".join(table)

    assert await cli(stack, OLGA, "--json", "approvals", "show", approval_id) == 0
    shown = json.loads(capsys.readouterr().out)
    assert (shown["id"], shown["tool"], shown["state"]) == (approval_id, "write_report", "pending")

    assert await cli(stack, OLGA, "approvals", "approve", approval_id, "--note", "ok") == 0
    assert capsys.readouterr().out.strip() == f"{approval_id}: approved"

    params = {"name": "write_report", "arguments": REPORT, "_meta": {META: approval_id}}
    done = (await reports.request("tools/call", params)).json()["result"]
    assert done["isError"] is False
    assert len(stack.log.of("write_report")) == 1

    assert await cli(stack, OLGA, "approvals", "list", "--state", "all") == 0
    assert "succeeded" in capsys.readouterr().out


async def test_deny_and_refusals_exit_1_with_the_reason_code(
    stack: Any, capsys: pytest.CaptureFixture
):
    _reports, approval_id = await held_write(stack)
    assert await cli(stack, BARTEK, "approvals", "approve", approval_id) == 1
    assert "operator_role_required (HTTP 403)" in capsys.readouterr().err
    assert await cli(stack, None, "approvals", "list") == 1
    assert "token_missing" in capsys.readouterr().err

    assert await cli(stack, ROOT, "--json", "approvals", "deny", approval_id) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "denied"
    assert await cli(stack, OLGA, "approvals", "approve", approval_id) == 1
    assert "approval_not_pending (HTTP 409)" in capsys.readouterr().err
    assert await cli(stack, OLGA, "approvals", "list") == 0
    assert capsys.readouterr().out.strip() == "no approvals"


async def test_kill_list_and_unkill(stack: Any, capsys: pytest.CaptureFixture):
    assert await cli(stack, ROOT, "kill", "nightly_etl", "--reason", "runaway") == 0
    assert capsys.readouterr().out.strip() == "nightly_etl: killed (0 unused approvals denied)"
    assert await cli(stack, ROOT, "kill") == 0
    listing = capsys.readouterr().out.splitlines()
    assert listing[1].split()[:3] == ["nightly_etl", "runaway", ROOT]
    assert await cli(stack, OLGA, "kill", "databot") == 1
    assert "admin_required" in capsys.readouterr().err
    assert await cli(stack, ROOT, "unkill", "nightly_etl") == 0
    assert capsys.readouterr().out.strip() == "nightly_etl: kill switch off"
    assert await cli(stack, ROOT, "--json", "kill") == 0
    assert json.loads(capsys.readouterr().out) == {"kills": []}


async def test_an_unreachable_operator_api_is_a_clean_error(capsys: pytest.CaptureFixture):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    code = await run(
        ["approvals", "list"],
        env={"ACL_OPERATOR_TOKEN": "t"},
        transport=httpx.MockTransport(refuse),
    )
    assert code == 1
    assert "unreachable" in capsys.readouterr().err


def test_every_registered_subcommand_is_reachable():
    parser = build_parser()
    for argv in (["approvals", "list"], ["kill", "a"], ["unkill", "a"]):
        assert callable(parser.parse_args(argv).run)
    with pytest.raises(SystemExit):
        parser.parse_args(["approvals"])  # a subcommand is required
