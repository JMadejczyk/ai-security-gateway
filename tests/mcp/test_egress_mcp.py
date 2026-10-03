"""``egress`` on ``/mcp/web`` (``adapter: http``): internal destinations never reach mcp-fetch.

The harness's `FakeResolver` (``stack.gateway.resolver``) stands in for DNS: every host is
public unless a test says otherwise, and no test queries real DNS.
"""

from typing import Any

import pytest
import yaml
from mcp_harness import MCPStack, connect, error_text

ANNA, ETL = "anna@demo", "svc:nightly_etl"
CLEAN_PAGE = "<html><body>Cables from 4.20 PLN/m.</body></html>"


def egress_verdicts(stack: MCPStack) -> list[dict[str, Any]]:
    entry = stack.gateway.audit_entries()[-1]
    return [v for v in entry["verdicts"] if v["control"] == "egress"]


@pytest.mark.control("egress", "deny")
async def test_the_metadata_endpoint_is_blocked_before_the_upstream(stack: MCPStack):
    web = await connect(stack, ANNA, "web")
    result = await web.call("fetch", url="http://169.254.169.254/latest/meta-data/")
    assert error_text(result) == "egress_private_address"
    assert stack.log.of("fetch") == []
    [verdict] = egress_verdicts(stack)
    assert (verdict["decision"], verdict["enforced"]) == ("block", True)
    session = await stack.gateway.container.sessions.get(
        stack.gateway.audit_entries()[-1]["session_id"]
    )
    assert session is not None
    assert session.risk == pytest.approx(0.3)


@pytest.mark.control("egress", "allow")
async def test_a_public_url_is_fetched(stack: MCPStack):
    stack.log.fetch_page = CLEAN_PAGE
    web = await connect(stack, ANNA, "web")
    result = await web.call("fetch", url="https://example.com/prices")
    assert result["isError"] is False, result
    assert [call.arguments["url"] for call in stack.log.of("fetch")] == [
        "https://example.com/prices"
    ]
    assert [v["reason_code"] for v in egress_verdicts(stack)] == ["egress_allowed"]
    assert stack.gateway.resolver.calls == [("example.com", 443)]


@pytest.mark.control("egress", "deny")
async def test_a_name_resolving_to_an_internal_address_is_blocked(stack: MCPStack):
    stack.gateway.resolver.answers["metadata.attacker.example"] = ["169.254.169.254"]
    web = await connect(stack, ANNA, "web")
    result = await web.call("fetch", url="https://metadata.attacker.example/")
    assert error_text(result) == "egress_private_address"
    assert stack.log.of("fetch") == []


@pytest.mark.control("egress", "deny")
async def test_an_unresolvable_host_is_blocked(stack: MCPStack):
    stack.gateway.resolver.answers["nowhere.example"] = []
    web = await connect(stack, ANNA, "web")
    assert error_text(await web.call("fetch", url="https://nowhere.example/")) == (
        "egress_unresolvable"
    )
    assert stack.log.of("fetch") == []


@pytest.mark.control("egress", "require_approval")
async def test_require_approval_mode_holds_the_fetch_for_an_operator(stack: MCPStack):
    path = stack.gateway.policy_path
    document = yaml.safe_load(path.read_text())
    document["controls"]["egress"] = {"mode": "require_approval"}
    path.write_text(yaml.safe_dump(document))
    assert stack.gateway.container.policy_store.reload().result == "ok"
    web = await connect(stack, ETL, "web")
    held = await web.call("fetch", url="http://10.0.0.5/admin")
    assert error_text(held).startswith("approval_required")
    assert "approval_id=apr-" in error_text(held)
    assert stack.log.of("fetch") == []
    [verdict] = egress_verdicts(stack)
    assert (verdict["decision"], verdict["reason_code"]) == (
        "require_approval",
        "egress_private_address",
    )
