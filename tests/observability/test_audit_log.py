"""The audit stream as Loki sees it: size-capped JSONL segments and policy-reload events."""

import fnmatch
import json
import os
import re
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

REPO_ROOT = Path(__file__).resolve().parents[2]

TS = datetime(2026, 10, 4, 10, 0, tzinfo=UTC)
SEGMENT = re.compile(r"audit-\d{8}T\d{12}Z-\d{6}\.jsonl")


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


def _write(path: Path, count: int, *, first: int = 0, max_bytes: int = 4096) -> None:
    with open(os.devnull, "w") as sink:
        logger = AuditLogger(stream=sink, path=path, max_bytes=max_bytes, backups=2)
        for index in range(first, first + count):
            logger.write(_entry(index))
        logger.close()


def _lines(files: list[Path]) -> list[dict[str, object]]:
    return [json.loads(line) for file in files for line in file.read_text().splitlines()]


def test_export_rolls_over_by_size_and_keeps_a_bounded_number_of_segments(
    tmp_path: Path,
) -> None:
    path = tmp_path / "audit" / "audit.jsonl"
    _write(path, 200)
    segments = sorted(path.parent.iterdir())
    assert len(segments) == 3  # the live one + backups=2; older ones deleted
    assert all(SEGMENT.fullmatch(p.name) for p in segments), [p.name for p in segments]
    assert all(p.stat().st_size <= 4096 for p in segments)
    lines = _lines(segments)  # name order is write order
    ids = [line["session_id"] for line in lines]
    assert ids == [f"s-{i}" for i in range(200 - len(ids), 200)]  # contiguous, newest last


def test_segments_are_never_renamed_or_rewritten(tmp_path: Path) -> None:
    """A tailer keys its offset by path: a path must keep naming the same bytes forever."""
    path = tmp_path / "audit.jsonl"
    with open(os.devnull, "w") as sink:
        logger = AuditLogger(stream=sink, path=path, max_bytes=4096, backups=50)
        seen: dict[str, bytes] = {}
        for index in range(300):
            logger.write(_entry(index))
            for segment in sorted(tmp_path.iterdir())[:-1]:  # every closed segment
                content = segment.read_bytes()
                assert seen.setdefault(segment.name, content) == content, segment.name
        logger.close()
    assert len(seen) > 5
    assert not path.exists()  # nothing is ever written under the base name itself


def test_a_restart_starts_a_new_segment_and_appends_nothing_to_old_ones(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    _write(path, 3, max_bytes=1 << 20)
    (first,) = sorted(tmp_path.iterdir())
    before = first.read_bytes()
    _write(path, 3, first=3, max_bytes=1 << 20)
    segments = sorted(tmp_path.iterdir())
    assert len(segments) == 2
    assert segments[0] == first
    assert first.read_bytes() == before
    assert [line["session_id"] for line in _lines(segments)] == [f"s-{i}" for i in range(6)]


def test_the_alloy_glob_matches_exactly_the_segments(tmp_path: Path) -> None:
    alloy = (REPO_ROOT / "observability" / "alloy" / "config.alloy").read_text()
    (pattern,) = re.findall(r'"__path__"\s*=\s*"([^"]+)"', alloy)
    assert Path(pattern).parent == Path("/var/log/acl")
    path = tmp_path / "audit.jsonl"
    _write(path, 50)
    names = [p.name for p in tmp_path.iterdir()]
    assert names
    assert all(fnmatch.fnmatch(name, Path(pattern).name) for name in names)
    assert not fnmatch.fnmatch("audit.jsonl", Path(pattern).name)


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
