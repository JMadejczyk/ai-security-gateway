"""``tool_poisoning`` on its own: the listing screen and the tools/call check, fake classifier."""

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest
from injection_kit import INJECT_MARKER, MarkerClassifier

from gateway.controls.tool_pinning import PinnedListing, PinStatus, listing_scope
from gateway.controls.tool_poisoning import (
    MAX_LISTED_TOOLS,
    ToolPoisoningControl,
    definition_text,
)
from gateway.core.envelope import Interaction
from gateway.core.types import Action, Channel, ControlMode, Decision, Stage
from gateway.injection.classifier import ClassifierRunner, InjectionScore, UnavailableClassifier
from gateway.policy.schema import ToolPoisoningConfig
from gateway.proxies.mcp import wire

BLOCK = ToolPoisoningConfig(mode=ControlMode.BLOCK)


def tool(name: str, description: str | None = None, **schema: Any) -> wire.ToolDefinition:
    document: dict[str, Any] = {
        "name": name,
        "inputSchema": {"type": "object", "properties": schema},
    }
    if description is not None:
        document["description"] = description
    return wire.ToolDefinition.model_validate(document)


CLEAN = tool("fetch", "Fetch a web page and return its text.", url={"type": "string"})
POISONED_DESCRIPTION = tool("add", f"Adds two numbers. {INJECT_MARKER}")
POISONED_PARAMETER = tool(
    "notes",
    "Saves a note.",
    text={"type": "string", "description": f"The note. {INJECT_MARKER}"},
)
POISONED_DEFAULT = tool(
    "search", "Search the docs.", q={"type": "string", "default": INJECT_MARKER}
)


def control(classifier: Any = None, *, budget_s: float = 10.0) -> ToolPoisoningControl:
    return ToolPoisoningControl(
        ClassifierRunner(classifier or MarkerClassifier()), listing_budget_s=budget_s
    )


def listing(*tools: wire.ToolDefinition) -> PinnedListing:
    return PinnedListing(
        server="web",
        pin=None,
        statuses={t.name: PinStatus.UNPINNED_ALLOWED for t in tools},
        unknown=PinStatus.NOT_PINNED,
        definitions=tools,
        schemas={t.name: t.input_schema for t in tools},
    )


@pytest.fixture
def call(make_ctx):
    def build(name: str) -> Interaction:
        return Interaction(
            session_id="s-test",
            principal="anna@demo",
            actor="databot",
            mode=make_ctx().mode,
            channel=Channel.MCP,
            action=Action.READ,
            resource="web:example.com",
            payload={"name": name, "arguments": {}},
            context=make_ctx(),
            server="web",
        )

    return build


# ------------------------------------------------------------------ listing screen

SCREEN_CASES = [
    ("clean-listing", [CLEAN], set()),
    ("poisoned-description", [CLEAN, POISONED_DESCRIPTION], {"add"}),
    ("poisoned-parameter-description", [CLEAN, POISONED_PARAMETER], {"notes"}),
    ("poisoned-default", [POISONED_DEFAULT, CLEAN], {"search"}),
    # A benign second entry must not stand in for a poisoned first one (or vice versa).
    ("duplicate-name", [tool("add", "Adds two numbers."), POISONED_DESCRIPTION], {"add"}),
]


@pytest.mark.parametrize(
    ("tools", "hidden"),
    [case[1:] for case in SCREEN_CASES],
    ids=[case[0] for case in SCREEN_CASES],
)
async def test_screen_hides_poisoned_tools(snapshot, tools, hidden):
    assert await control().screen_listing("web", tools, snapshot) == hidden


async def test_screen_uses_the_configured_threshold(snapshot_from, policy_doc):
    classifier = MarkerClassifier({"numbers": 0.9})
    tools = [tool("add", "Adds two numbers.")]
    assert await control(classifier).screen_listing("web", tools, snapshot_from(policy_doc)) == {
        "add"
    }
    policy_doc["controls"]["tool_poisoning"] = {"threshold": 0.95}
    relaxed = snapshot_from(policy_doc)
    assert await control(classifier).screen_listing("web", tools, relaxed) == set()


async def test_screen_fails_closed_when_the_classifier_is_off(snapshot):
    screen = ToolPoisoningControl(ClassifierRunner(UnavailableClassifier()))
    assert await screen.screen_listing("web", [CLEAN, POISONED_DESCRIPTION], snapshot) == {
        "fetch",
        "add",
    }


async def test_screen_hides_what_it_could_not_classify_in_time(snapshot):
    def slow(texts: Sequence[str]) -> list[InjectionScore]:
        import time  # noqa: PLC0415 -- local to the one slow fake

        time.sleep(0.3)
        return [InjectionScore(0.0, 0, 0) for _ in texts]

    started = asyncio.get_running_loop().time()
    hidden = await control(slow, budget_s=0.05).screen_listing("web", [CLEAN], snapshot)
    assert hidden == {"fetch"}
    assert asyncio.get_running_loop().time() - started < 0.25


async def test_screen_hides_tools_past_the_listing_cap(snapshot):
    tools = [tool(f"t{i}", "A tool.") for i in range(MAX_LISTED_TOOLS + 3)]
    hidden = await control().screen_listing("web", tools, snapshot)
    assert hidden == {f"t{i}" for i in range(MAX_LISTED_TOOLS, MAX_LISTED_TOOLS + 3)}


async def test_a_listing_is_classified_once(snapshot):
    classifier = MarkerClassifier()
    screen = control(classifier)
    for _ in range(3):
        await screen.screen_listing("web", [CLEAN, POISONED_DESCRIPTION], snapshot)
    assert len(classifier.calls) == 1


# -------------------------------------------------------------------- tools/call

CALL_CASES = [
    ("clean-tool", [CLEAN], "fetch", Decision.ALLOW, "no_tool_poisoning"),
    ("poisoned-tool-called-by-name", [CLEAN, POISONED_DESCRIPTION], "add", Decision.BLOCK,
     "tool_poisoning_detected"),
    ("poisoned-parameter", [POISONED_PARAMETER], "notes", Decision.BLOCK,
     "tool_poisoning_detected"),
    ("not-advertised", [CLEAN], "exec_shell", Decision.ALLOW, "tool_not_advertised"),
]  # fmt: skip


@pytest.mark.parametrize(
    ("tools", "name", "decision", "reason_code"),
    [case[1:] for case in CALL_CASES],
    ids=[case[0] for case in CALL_CASES],
)
async def test_call_checks_the_definition_the_caller_was_shown(
    call, tools, name, decision, reason_code
):
    with listing_scope(listing(*tools)):
        verdict = await control().evaluate(call(name), Stage.PRE, BLOCK)
    assert (verdict.decision, verdict.enforced, verdict.reason_code) == (
        decision,
        True,
        reason_code,
    )
    if decision is Decision.BLOCK:
        assert verdict.reason == "classifier score 0.99-1.00"


async def test_a_call_without_a_verified_listing_fails_closed(call):
    verdict = await control().evaluate(call("fetch"), Stage.PRE, BLOCK)
    assert (verdict.decision, verdict.reason_code) == (
        Decision.BLOCK,
        "tool_definition_unavailable",
    )


async def test_a_call_fails_closed_when_the_classifier_is_off(call):
    screen = ToolPoisoningControl(ClassifierRunner(UnavailableClassifier()))
    with listing_scope(listing(CLEAN)):
        verdict = await screen.evaluate(call("fetch"), Stage.PRE, BLOCK)
    assert (verdict.decision, verdict.reason_code) == (Decision.BLOCK, "classifier_unavailable")


def test_definition_text_covers_every_prose_field_but_not_grammar():
    definition = wire.ToolDefinition.model_validate(
        {
            "name": "notes",
            "title": "Notes",
            "description": "Saves a note.",
            "inputSchema": {
                "type": "object",
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "required": ["text"],
                "properties": {
                    "text": {
                        "type": "string",
                        "format": "uri",
                        "description": "The note body.",
                        "examples": ["buy milk"],
                    }
                },
            },
        }
    )
    lines = definition_text(definition).splitlines()
    assert sorted(lines) == sorted(
        ["notes", "Notes", "Saves a note.", "text", "The note body.", "buy milk"]
    )


# ------------------------------------------------- codex review fixes (2026-10-03)

DATA_VALUE_CASES = [
    ("default-object", tool("cfg", "Configure.", opts={"type": "object",
                                                        "default": {"type": INJECT_MARKER}})),
    ("examples-object", tool("link", "Link.", ref={"type": "string",
                                                   "examples": [{"uri": INJECT_MARKER}]})),
    ("enum-value", tool("mode", "Mode.", m={"type": "string", "enum": ["a", INJECT_MARKER]})),
    ("const-value", tool("fixed", "Fixed.", c={"const": {"format": INJECT_MARKER}})),
    ("object-key-in-default", tool("keys", "Keys.", k={"default": {INJECT_MARKER: 1}})),
]  # fmt: skip


@pytest.mark.parametrize(
    "definition",
    [case[1] for case in DATA_VALUE_CASES],
    ids=[case[0] for case in DATA_VALUE_CASES],
)
async def test_instructions_inside_schema_data_values_are_classified(snapshot, call, definition):
    """Codex P1 #4: keyword skipping (``type``, ``uri``, ``format``) leaked into data values."""
    assert await control().screen_listing("web", [definition], snapshot) == {definition.name}
    with listing_scope(listing(definition)):
        verdict = await control().evaluate(call(definition.name), Stage.PRE, BLOCK)
    assert verdict.reason_code == "tool_poisoning_detected"


async def test_a_duplicate_past_the_listing_cap_hides_its_name(snapshot):
    """Codex P2 #6: 256 benign ``notes`` hid the unclassified, poisoned 257th."""
    tools = [tool("notes", "Saves a note.") for _ in range(MAX_LISTED_TOOLS)]
    tools.append(tool("notes", f"Saves a note. {INJECT_MARKER}"))
    assert await control().screen_listing("web", tools, snapshot) == {"notes"}
