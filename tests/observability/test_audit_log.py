"""The audit stream as Loki sees it: size-rotated JSONL export and policy-reload events."""

import json
from datetime import UTC, datetime
from pathlib import Path

import yaml
from gateway_testkit import Harness, bearer

from gateway.core.types import Channel, Decision
from gateway.telemetry import (
    AuditEntry,
    AuditLatency,
    AuditLogger,
    PolicyReloadEvent,
    ReloadResult,
)

TS = datetime(2026, 10, 4, 10, 0, tzinfo=UTC)


def _entry(index: int) -> AuditEntry:
    return AuditEntry(
        ts=TS,
        session_id=f"s-{index}",
        channel=Channel.MCP,
        decision=Decision.ALLOW,
        reason_code="allowed",
        status=200,
        policy_revision="a1c9e2f04b7d",
        latency_ms=AuditLatency(total=1.0),
    )


def test_export_rotates_by_size_and_keeps_a_bounded_number_of_files(tmp_path: Path) -> None:
    path = tmp_path / "audit" / "audit.jsonl"
    with open("/dev/null", "w") as sink:
        logger = AuditLogger(stream=sink, path=path, max_bytes=4096, backups=2)
        for index in range(200):
            logger.write(_entry(index))
        logger.close()
    files = sorted(p.name for p in path.parent.iterdir())
    assert files == ["audit.jsonl", "audit.jsonl.1", "audit.jsonl.2"]
    for file in path.parent.iterdir():
        assert file.stat().st_size <= 4096
        lines = file.read_text().splitlines()
        assert all(json.loads(line)["policy_revision"] == "a1c9e2f04b7d" for line in lines)
    newest = json.loads(path.read_text().splitlines()[-1])
    assert newest["session_id"] == "s-199"  # the live file holds the latest lines


def test_reload_event_serializes_like_an_audit_line() -> None:
    event = PolicyReloadEvent(ts=TS, result=ReloadResult.OK, revision="b2", previous_revision="a1")
    assert json.loads(event.model_dump_json()) == {
        "ts": "2026-10-04T10:00:00.000Z",
        "event": "policy_reload",
        "result": "ok",
        "revision": "b2",
        "previous_revision": "a1",
    }


async def test_every_reload_attempt_lands_in_the_audit_stream(gateway: Harness) -> None:
    admin = bearer(await gateway.operator_token("root@demo"))
    before = gateway.container.policy_store.current.revision

    unchanged = await gateway.operator.post("/admin/reload", headers=admin)
    document = yaml.safe_load(gateway.policy_path.read_text())
    document["controls"]["sql_guard"]["max_cost"] += 1
    gateway.policy_path.write_text(yaml.safe_dump(document))
    changed = await gateway.operator.post("/admin/reload", headers=admin)
    gateway.policy_path.write_text("schema_version: [not a policy")
    invalid = await gateway.operator.post("/admin/reload", headers=admin)

    assert [r.json()["result"] for r in (unchanged, changed, invalid)] == [
        "unchanged",
        "ok",
        "invalid",
    ]
    after = changed.json()["revision"]
    events = gateway.audit_events()
    assert [(e["result"], e["previous_revision"], e["revision"]) for e in events] == [
        ("unchanged", before, before),
        ("ok", before, after),
        ("invalid", after, after),
    ]
    assert all(e["event"] == "policy_reload" for e in events)
    # The loader's error text stays out of the audit stream (no free text there).
    assert all(set(e) == {"ts", "event", "result", "revision", "previous_revision"} for e in events)
    assert gateway.audit_entries() == []  # reloads are not decisions
