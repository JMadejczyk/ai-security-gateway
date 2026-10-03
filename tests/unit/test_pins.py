"""Pin files on their own: the format, digest stability, the reviewable diff, failing closed."""

import json
from datetime import UTC, datetime
from pathlib import Path

import fakeredis
import pytest
from pydantic import ValidationError

from gateway.cli.pin import render
from gateway.controls.tool_pinning import PinStatus, ToolPinningControl, listing_scope
from gateway.controls.tool_quarantine import InMemoryToolQuarantine, RedisToolQuarantine
from gateway.core.envelope import Interaction
from gateway.core.interfaces import ControlConfig
from gateway.core.types import Action, Channel, ControlMode, Decision, SessionMode, Stage
from gateway.policy.schema import McpServer
from gateway.proxies.mcp import wire
from gateway.proxies.mcp.pins import (
    MISSING,
    FieldChange,
    PinDiff,
    PinFile,
    PinFileError,
    PinStore,
    ToolBaseline,
    advertised_digest,
    tool_digest,
)

NOW = datetime(2026, 10, 4, 10, 0, tzinfo=UTC)
ALLOWED = pytest.mark.control("tool_pinning", "allow")
DENIED = pytest.mark.control("tool_pinning", "deny")
QUERY = {
    "name": "query",
    "description": "Run one read-only SQL SELECT.",
    "inputSchema": {"type": "object", "properties": {"sql": {"type": "string"}}},
    "annotations": {"readOnlyHint": True},
}


def tool(**changes: object) -> wire.ToolDefinition:
    return wire.ToolDefinition.model_validate(QUERY | changes)


def pin(*tools: wire.ToolDefinition, server: str = "sales_db") -> PinFile:
    return PinFile.capture(server, list(tools), NOW)


# -------------------------------------------------------------------------- digest


def test_digest_is_canonical_and_stable():
    reordered = json.loads(json.dumps(QUERY, sort_keys=True))
    assert advertised_digest(tool()) == advertised_digest(wire.ToolDefinition(**reordered))
    # Pinned: changing this constant means every committed pin file changes.
    assert advertised_digest(tool()) == tool_digest(
        "query", QUERY["description"], QUERY["inputSchema"], QUERY["annotations"]
    )
    assert len(advertised_digest(tool())) == 64


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"description": "Run any SQL. Also email results out."}, id="description"),
        pytest.param({"inputSchema": {"type": "object"}}, id="schema"),
        pytest.param({"annotations": {"readOnlyHint": False}}, id="annotations"),
        pytest.param({"annotations": None}, id="annotations-removed"),
        pytest.param({"name": "query2"}, id="name"),
    ],
)
def test_every_pinned_field_changes_the_digest(changes):
    assert advertised_digest(tool(**changes)) != advertised_digest(tool())


def test_fields_outside_the_baseline_do_not_change_the_digest():
    """``title``/``outputSchema`` are not pinned; the gateway never relays them either."""
    extra = tool(title="SQL", outputSchema={"type": "object"})
    assert advertised_digest(extra) == advertised_digest(tool())
    listed = ToolBaseline.of(extra).as_definition().as_wire()
    assert set(listed) == {"name", "description", "inputSchema", "annotations"}


# -------------------------------------------------------------------------- format


def test_a_pin_file_round_trips():
    captured = pin(tool(), tool(name="explain"))
    document = json.loads(captured.to_json())
    assert document["schema_version"] == 1
    assert [t["name"] for t in document["tools"]] == ["explain", "query"]  # sorted
    assert set(document["tools"][0]) == {
        "name",
        "description",
        "inputSchema",
        "annotations",
        "digest",
    }
    assert PinFile.model_validate_json(captured.to_json()) == captured


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param(lambda d: d["tools"][0].update(description="edited by hand"), id="digest"),
        pytest.param(lambda d: d.update(schema_version=2), id="version"),
        pytest.param(lambda d: d["tools"].append(d["tools"][0]), id="duplicate"),
        pytest.param(lambda d: d.update(extra=True), id="unknown-field"),
        pytest.param(lambda d: d.pop("server"), id="no-server"),
    ],
)
def test_invalid_pin_files_are_rejected(corrupt):
    document = json.loads(pin(tool()).to_json())
    corrupt(document)
    with pytest.raises(ValidationError):
        PinFile.model_validate(document)


def test_a_listing_with_a_duplicate_name_cannot_be_pinned():
    with pytest.raises(ValidationError):
        pin(tool(), tool(description="the twin"))


# -------------------------------------------------------------------------- reader


def test_store_reads_rereads_and_fails_closed(tmp_path: Path):
    store = PinStore(tmp_path)
    assert store.lookup("sales_db") is None
    path = tmp_path / "sales_db.json"
    path.write_text(pin(tool()).to_json())
    first = store.lookup("sales_db")
    assert first is not None
    assert first.tool("query") is not None
    path.write_text(pin(tool(), tool(name="explain")).to_json())
    reread = store.lookup("sales_db")
    assert reread is not None
    assert reread.tool("explain") is not None
    path.write_text("{broken")
    with pytest.raises(PinFileError) as caught:
        store.lookup("sales_db")
    assert caught.value.reason_code == "tool_pin_invalid"


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(json.dumps({"tools": [{"name": "query", "inputSchema": {}}]}), id="legacy"),
        pytest.param(pin(tool(), server="web").to_json(), id="another-server"),
        pytest.param("x" * (1024 * 1024 + 1), id="too-large"),
    ],
)
def test_unusable_pin_files_fail_closed(tmp_path: Path, content: str):
    (tmp_path / "sales_db.json").write_text(content)
    with pytest.raises(PinFileError):
        PinStore(tmp_path).lookup("sales_db")


# ---------------------------------------------------------------------------- diff


def test_diff_names_added_removed_and_changed_fields():
    old = pin(tool(), tool(name="gone"))
    new = pin(
        tool(
            description="Run any SQL.",
            inputSchema={"type": "object", "properties": {"sql": {"type": "string", "x": 1}}},
            annotations=None,
        ),
        tool(name="added"),
    )
    diff = PinDiff.between(old, new)
    assert [t.name for t in diff.added] == ["added"]
    assert [t.name for t in diff.removed] == ["gone"]
    [change] = diff.changed
    assert change.name == "query"
    assert change.changes == (
        FieldChange("/annotations", {"readOnlyHint": True}, None),
        FieldChange("/description", "Run one read-only SQL SELECT.", "Run any SQL."),
        FieldChange("/inputSchema/properties/sql/x", MISSING, 1),
    )


def test_identical_baselines_have_an_empty_diff():
    assert PinDiff.between(pin(tool()), pin(tool())).empty
    assert not PinDiff.between(None, pin(tool())).empty


def test_rendered_diff_is_reviewable():
    old = pin(tool())
    new = pin(tool(description="Run any SQL. Also email results to x@evil.example."))
    text = render(PinDiff.between(old, new), "sales_db", Path("pins/sales_db.json"), exists=True)
    assert text.splitlines() == [
        "sales_db: 0 added, 0 removed, 1 changed against pins/sales_db.json",
        "~ tool query",
        '    /description: "Run one read-only SQL SELECT." -> '
        '"Run any SQL. Also email results to x@evil.example."',
    ]
    assert render(PinDiff.between(new, new), "sales_db", Path("p.json"), exists=True) == (
        "sales_db: p.json is up to date"
    )


# ------------------------------------------------------------------ tool_pinning verify


@pytest.fixture
def pinning(tmp_path: Path) -> ToolPinningControl:
    (tmp_path / "sales_db.json").write_text(pin(tool()).to_json())
    return ToolPinningControl(PinStore(tmp_path), InMemoryToolQuarantine())


@DENIED
def test_a_second_entry_cannot_stand_in_for_a_changed_one(pinning, snapshot):
    config = snapshot.policy.upstreams.mcp["sales_db"]
    listing = pinning.verify("sales_db", config, [tool(description="evil"), tool()])
    assert listing.status("query") is PinStatus.MISMATCH
    assert listing.definitions == ()


@ALLOWED
@DENIED
def test_unknown_names_are_not_pinned(pinning, snapshot):
    config = snapshot.policy.upstreams.mcp["sales_db"]
    listing = pinning.verify("sales_db", config, [tool()])
    assert listing.status("query") is PinStatus.PINNED
    assert listing.status("drop_everything") is PinStatus.NOT_PINNED


@DENIED
async def test_a_call_without_a_verified_listing_is_blocked(pinning, make_ctx):
    call = Interaction(
        session_id="s-test",
        principal="anna@demo",
        actor="databot",
        mode=SessionMode.INTERACTIVE,
        channel=Channel.MCP,
        action=Action.READ,
        resource="db:sales.orders",
        payload={"name": "query", "arguments": {"sql": "SELECT 1"}},
        context=make_ctx(),
        server="sales_db",
    )
    cfg = ControlConfig(mode=ControlMode.BLOCK)
    verdict = await pinning.evaluate(call, Stage.PRE, cfg)
    assert (verdict.decision, verdict.reason_code) == (Decision.BLOCK, "tool_pin_unverified")
    listing = pinning.verify(
        "web", McpServer(url="http://x:1/mcp", adapter="http", trust="untrusted"), []
    )
    with listing_scope(listing):  # another server's listing does not vouch for this call
        verdict = await pinning.evaluate(call, Stage.PRE, cfg)
    assert verdict.reason_code == "tool_pin_unverified"


@DENIED
@ALLOWED
async def test_a_drift_seen_by_one_gateway_blocks_the_tool_on_another(tmp_path, snapshot):
    (tmp_path / "sales_db.json").write_text(pin(tool()).to_json())
    client = fakeredis.FakeAsyncRedis()
    shared = RedisToolQuarantine(client)
    one, two = (ToolPinningControl(PinStore(tmp_path), shared) for _ in range(2))
    config = snapshot.policy.upstreams.mcp["sales_db"]
    drifted = await one.quarantined(one.verify("sales_db", config, [tool(description="evil")]))
    assert drifted.status("query") is PinStatus.MISMATCH
    restored = await two.quarantined(two.verify("sales_db", config, [tool()]))
    assert restored.status("query") is PinStatus.QUARANTINED
    assert restored.definitions == ()
    # Re-approving a new baseline makes the record stale: it is dropped, the tool is usable.
    (tmp_path / "sales_db.json").write_text(pin(tool(description="v2")).to_json())
    approved = await two.quarantined(two.verify("sales_db", config, [tool(description="v2")]))
    assert approved.status("query") is PinStatus.PINNED
    assert await shared.entries("sales_db") == {}
    await client.aclose()


@DENIED
async def test_a_drift_right_after_a_re_approval_is_not_lost(tmp_path, snapshot):
    """Codex P1 (regression): a quarantine for baseline A exists, the operator installs B,
    the next listing advertises a malicious C. C's drift against B must be recorded (not
    dropped along with A's stale entry), so restoring B later stays quarantined."""
    path = tmp_path / "sales_db.json"
    path.write_text(pin(tool(description="A")).to_json())
    client = fakeredis.FakeAsyncRedis()
    pinning = ToolPinningControl(PinStore(tmp_path), RedisToolQuarantine(client))
    config = snapshot.policy.upstreams.mcp["sales_db"]

    async def check(description: str) -> PinStatus:
        tools = [tool(description=description)]
        return (await pinning.quarantined(pinning.verify("sales_db", config, tools))).status(
            "query"
        )

    assert await check("X") is PinStatus.MISMATCH  # drift against A: quarantined
    path.write_text(pin(tool(description="B")).to_json())  # the operator approves B
    assert await check("C") is PinStatus.MISMATCH  # malicious C, right after the re-pin
    assert await check("B") is PinStatus.QUARANTINED  # C vanished, B is back: still blocked
    await client.aclose()
