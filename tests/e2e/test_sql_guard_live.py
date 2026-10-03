"""``sql_guard`` live: demo step 4, the forced LIMIT, and the limits mcp-postgres applies.

Agent calls run inside the ``agent`` container through the gateway (`test_mcp_live.call_tool`).
Whether a statement ran is read from Postgres itself: ``pg_stat_user_tables`` counts the scans
of ``sales.payments``, which only a statement reading that table causes (``EXPLAIN`` does not).
"""

import json
import os
import subprocess
import time

import pytest
from live_stack import REPO_ROOT, LiveStack
from test_mcp_live import call_tool

pytestmark = pytest.mark.docker

ANNA = "anna@demo"
HEAVY = (
    "SELECT COUNT(*) FROM sales.customers c CROSS JOIN sales.orders o CROSS JOIN sales.payments p"
)
PAYMENT_SCANS = (
    "SELECT coalesce(seq_scan, 0) + coalesce(idx_scan, 0) FROM pg_stat_user_tables "
    "WHERE relid = 'sales.payments'::regclass"
)
STATS_FLUSH_S = 20.0  # backends flush table statistics when idle, within ~10 s

# Runs in the mcp-postgres container, which holds ACL_INTERNAL_KEY: calls `query` directly with
# a principal assertion whose limits claim says 100 ms. argv: SQL. Prints the tools/call result.
_DIRECT_QUERY = """
import json, os, sys, time
import httpx, jwt
now = int(time.time())
token = jwt.encode({"iss": "ai-control-layer", "aud": "mcp-postgres", "sub": "anna@demo",
                    "iat": now, "exp": now + 30,
                    "limits": {"stmt_timeout_ms": 100, "max_rows": 500,
                               "max_result_bytes": 1048576}},
                   os.environ["ACL_INTERNAL_KEY"], algorithm="HS256")
url = "http://127.0.0.1:8000/mcp"
headers = {"accept": "application/json, text/event-stream", "content-type": "application/json",
           "x-acl-principal": token}
def rpc(message):
    reply = client.post(url, headers=headers, json=message)
    reply.raise_for_status()
    if reply.headers.get("mcp-session-id"):
        headers["mcp-session-id"] = reply.headers["mcp-session-id"]
    if not reply.text:
        return None
    text = reply.text
    if reply.headers["content-type"].startswith("text/event-stream"):
        text = [l[5:].strip() for l in text.splitlines() if l.startswith("data:")][-1]
    return json.loads(text)
with httpx.Client(timeout=30) as client:
    rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "e2e", "version": "0"}}})
    headers["mcp-protocol-version"] = "2025-06-18"
    rpc({"jsonrpc": "2.0", "method": "notifications/initialized"})
    answer = rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                  "params": {"name": "query", "arguments": {"sql": sys.argv[1]}}})
print(json.dumps(answer["result"]))
"""


def payment_scans(stack: LiveStack) -> int:
    completed = subprocess.run(  # noqa: S603 -- fixed argv, docker resolved from PATH
        [
            stack.docker,
            "compose",
            "exec",
            "-T",
            "postgres",
            "psql",
            "--no-psqlrc",
            "-U",
            "postgres",
            "-d",
            "acl_demo",
            "-tAc",
            PAYMENT_SCANS,
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
        env=dict(os.environ),
        timeout=60,
    )
    return int(completed.stdout.strip())


def wait_for_scans_above(stack: LiveStack, baseline: int) -> int:
    deadline = time.monotonic() + STATS_FLUSH_S
    while (scans := payment_scans(stack)) <= baseline:
        assert time.monotonic() < deadline, "sales.payments was never scanned"
        time.sleep(1)
    return scans


def text_of(result: dict) -> str:
    return result["content"][0]["text"]


def test_heavy_cross_join_is_refused_after_explain_and_never_runs(live_stack: LiveStack):
    # Positive control: an allowed statement on sales.payments does move the counter.
    before = payment_scans(live_stack)
    allowed = call_tool(live_stack, ANNA, "query", sql="SELECT COUNT(*) FROM sales.payments")
    assert allowed["isError"] is False, allowed
    settled = wait_for_scans_above(live_stack, before)

    refused = call_tool(live_stack, ANNA, "query", sql=HEAVY)
    assert refused["isError"] is True
    assert text_of(refused) == "sql_cost_exceeded"
    time.sleep(STATS_FLUSH_S)
    assert payment_scans(live_stack) == settled


def test_select_star_returns_at_most_force_limit_rows(live_stack: LiveStack):
    result = call_tool(live_stack, ANNA, "query", sql="SELECT * FROM sales.orders")
    assert result["isError"] is False, result
    rows = result["structuredContent"]["result"]
    assert len(rows) == 500  # anna sees 800 orders; policy force_limit is 500


def test_agents_cannot_call_the_planner_tool(live_stack: LiveStack):
    result = call_tool(live_stack, ANNA, "explain", sql="SELECT * FROM sales.orders LIMIT 5")
    assert result["isError"] is True
    assert text_of(result) == "tool_not_mapped"


def test_mcp_postgres_applies_the_signed_statement_timeout(live_stack: LiveStack):
    output = live_stack.run_in("mcp-postgres", _DIRECT_QUERY, HEAVY + " LIMIT 500", env={})
    result = json.loads(output)
    assert result["isError"] is True, result
    assert text_of(result).endswith("statement canceled: execution limit exceeded")
