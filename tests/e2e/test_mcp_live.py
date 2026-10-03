"""Demo step 1, live: the same question through the gateway, different principal, different rows.

The MCP client runs inside the ``agent`` container, which can reach only the gateway
(``http://gateway:8080``). The gateway forwards the principal to mcp-postgres in
``X-ACL-Principal``; Postgres row-level security does the rest.
"""

import json

import pytest
from live_stack import LiveStack

pytestmark = pytest.mark.docker

GATEWAY = "http://gateway:8080/mcp/sales_db"

# Runs in the agent container (python:3.12 + httpx). The bearer token arrives in $ACL_TOKEN.
# argv: url, tool, JSON arguments. Prints the tools/call result as JSON.
_CALL_TOOL = """
import json, os, sys
import httpx
url, tool, arguments = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
headers = {
    "authorization": "Bearer " + os.environ["ACL_TOKEN"],
    "accept": "application/json, text/event-stream",
    "content-type": "application/json",
}
with httpx.Client(timeout=60) as client:
    init = client.post(url, headers=headers, json={
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                   "clientInfo": {"name": "e2e", "version": "0"}}})
    init.raise_for_status()
    headers["mcp-session-id"] = init.headers["mcp-session-id"]
    headers["mcp-protocol-version"] = init.json()["result"]["protocolVersion"]
    initialized = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    client.post(url, headers=headers, json=initialized).raise_for_status()
    reply = client.post(url, headers=headers, json={
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": tool, "arguments": arguments}})
    reply.raise_for_status()
    client.delete(url, headers=headers)
print(json.dumps(reply.json()["result"]))
"""


def call_tool(stack: LiveStack, sub: str, tool: str, **arguments: object) -> dict:
    token = stack.token(sub)
    output = stack.run_in(
        "agent", _CALL_TOOL, GATEWAY, tool, json.dumps(arguments), env={"ACL_TOKEN": token}
    )
    return json.loads(output)


def customer_count(result: dict) -> int:
    assert result["isError"] is False, result
    [row] = result["structuredContent"]["result"]
    return row["count"]


@pytest.mark.control("authz", "allow")
def test_same_count_query_sees_different_rows_per_principal(live_stack: LiveStack):
    sql = "SELECT COUNT(*) AS count FROM sales.customers"
    anna = customer_count(call_tool(live_stack, "anna@demo", "query", sql=sql))
    bartek = customer_count(call_tool(live_stack, "bartek@demo", "query", sql=sql))
    assert (anna, bartek) == (40, 7)


@pytest.mark.control("authz", "deny")
def test_intern_is_refused_the_payments_table(live_stack: LiveStack):
    result = call_tool(
        live_stack, "bartek@demo", "query", sql="SELECT COUNT(*) FROM sales.payments"
    )
    assert result["isError"] is True
    assert result["content"][0]["text"] == "outside_principal_scope"
