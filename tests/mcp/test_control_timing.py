"""Per-control time and verdict counts, as the audit entry and the metrics report them.

- The audit entry's ``latency_ms.controls`` is the time each control took over the whole call,
  every stage summed: a control running pre and post reports both runs, and agrees with what
  ``acl_control_latency_seconds`` observed for the call.
- A verdict the pipeline attaches to every interaction of a call (``budget`` reserves once per
  call) is counted and timed once, not once per interaction.
- ``authz`` (base authorization plus session restrictions) is timed like any control.
"""

from typing import Final

import pytest
from mcp_harness import MCPStack, connect

from gateway.telemetry import CONTROL_LATENCY, CONTROL_VERDICTS

ANNA: Final = "anna@demo"
ONE_TABLE: Final = "SELECT id, amount FROM sales.orders"
TWO_TABLES: Final = (
    "SELECT c.name FROM sales.customers c JOIN sales.orders o ON o.customer_id = c.id"
)
BOTH_STAGES: Final = ("pii", "secrets", "signatures")  # each runs on the arguments and the result


def _latency() -> tuple[dict[str, float], dict[str, float]]:
    """acl_control_latency_seconds per control: (sum in ms, observation count)."""
    sums: dict[str, float] = {}
    counts: dict[str, float] = {}
    for metric in CONTROL_LATENCY.collect():
        for sample in metric.samples:
            control = sample.labels["control"]
            if sample.name.endswith("_sum"):
                sums[control] = sample.value * 1000
            elif sample.name.endswith("_count"):
                counts[control] = sample.value
    return sums, counts


def _verdict_counts() -> dict[str, float]:
    """acl_control_verdicts_total per control, every decision summed."""
    totals: dict[str, float] = {}
    for metric in CONTROL_VERDICTS.collect():
        for sample in metric.samples:
            if sample.name.endswith("_total"):
                control = sample.labels["control"]
                totals[control] = totals.get(control, 0.0) + sample.value
    return totals


def _delta(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    return {k: v - before.get(k, 0.0) for k, v in after.items() if v != before.get(k, 0.0)}


async def _query(stack: MCPStack, sql: str) -> tuple[list[dict], dict, dict, dict]:
    """Run ``sql`` once; its audit entries and the metric deltas it caused."""
    sales = await connect(stack, ANNA, "sales_db")
    seen = len(stack.gateway.audit_entries())
    sums, counts = _latency()
    verdicts = _verdict_counts()
    assert (await sales.call("query", sql=sql))["isError"] is False
    after_sums, after_counts = _latency()
    entries = [e for e in stack.gateway.audit_entries()[seen:] if e["channel"] == "mcp"]
    return (
        entries,
        _delta(after_sums, sums),
        _delta(after_counts, counts),
        _delta(_verdict_counts(), verdicts),
    )


async def test_audit_control_time_sums_every_stage_and_matches_the_histogram(stack: MCPStack):
    [entry], sums, counts, _ = await _query(stack, ONE_TABLE)
    audited = entry["latency_ms"]["controls"]
    for control in BOTH_STAGES:
        assert counts[control] == 2, control  # one observation per stage
    assert set(BOTH_STAGES) <= set(sums)
    for control, total_ms in sums.items():
        assert audited[control] == pytest.approx(total_ms, rel=1e-6, abs=1e-9), control


async def test_a_call_wide_verdict_is_counted_once_on_a_two_table_query(stack: MCPStack):
    entries, sums, counts, verdicts = await _query(stack, TWO_TABLES)
    assert len(entries) == 2  # one interaction (and audit entry) per table
    assert verdicts["budget"] == 1  # one reservation for the call
    assert counts["budget"] == 1
    assert verdicts["authz"] == 2  # each table is authorized on its own
    assert counts["authz"] == 2
    for entry in entries:  # both entries carry the call's latency once, budget included
        assert entry["latency_ms"]["controls"]["budget"] == pytest.approx(sums["budget"])


async def test_authz_is_timed(stack: MCPStack):
    [entry], sums, _, _ = await _query(stack, ONE_TABLE)
    assert entry["latency_ms"]["controls"]["authz"] > 0
    assert sums["authz"] > 0
