"""``tool_poisoning`` on its own: the listing screen and the tools/call check, fake classifier."""

import asyncio
import copy
import json
from collections.abc import Sequence
from typing import Any, cast

import pytest
from injection_kit import DOUBT_MARKER, INJECT_MARKER, MarkerClassifier, ScriptedJudge
from judge_kit import set_judges

from gateway.controls.prompt_injection import InjectionJudgement
from gateway.controls.tool_pinning import PinnedListing, PinStatus, listing_scope
from gateway.controls.tool_poisoning import (
    MAX_CONCURRENT_JUDGES,
    MAX_LISTED_TOOLS,
    ToolPoisoningControl,
    definition_text,
)
from gateway.core.envelope import Interaction
from gateway.core.types import Action, Channel, ControlMode, Decision, Stage
from gateway.injection.classifier import ClassifierRunner, InjectionScore, UnavailableClassifier
from gateway.judges.client import JudgeClient, JudgeResult
from gateway.policy.loader import PolicyLoadError
from gateway.policy.schema import ToolPoisoningConfig
from gateway.proxies.mcp import wire
from gateway.upstream import Upstream, UpstreamResult

BLOCK = ToolPoisoningConfig(mode=ControlMode.BLOCK)
ALLOWED = pytest.mark.control("tool_poisoning", "allow")
DENIED = pytest.mark.control("tool_poisoning", "deny")


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
    [pytest.param(*case[1:], marks=DENIED if case[2] else ALLOWED) for case in SCREEN_CASES],
    ids=[case[0] for case in SCREEN_CASES],
)
async def test_screen_hides_poisoned_tools(snapshot, tools, hidden):
    assert await control().screen_listing("web", tools, snapshot) == hidden


@DENIED
@ALLOWED
async def test_screen_uses_the_configured_threshold(snapshot_from, policy_doc):
    classifier = MarkerClassifier({"numbers": 0.9})
    tools = [tool("add", "Adds two numbers.")]
    assert await control(classifier).screen_listing("web", tools, snapshot_from(policy_doc)) == {
        "add"
    }
    policy_doc["controls"]["tool_poisoning"] = {"threshold": 0.95}
    relaxed = snapshot_from(policy_doc)
    assert await control(classifier).screen_listing("web", tools, relaxed) == set()


@DENIED
async def test_screen_fails_closed_when_the_classifier_is_off(snapshot):
    screen = ToolPoisoningControl(ClassifierRunner(UnavailableClassifier()))
    assert await screen.screen_listing("web", [CLEAN, POISONED_DESCRIPTION], snapshot) == {
        "fetch",
        "add",
    }


@DENIED
async def test_screen_hides_what_it_could_not_classify_in_time(snapshot):
    def slow(texts: Sequence[str]) -> list[InjectionScore]:
        import time  # noqa: PLC0415 -- local to the one slow fake

        time.sleep(0.3)
        return [InjectionScore(0.0, 0, 0) for _ in texts]

    started = asyncio.get_running_loop().time()
    hidden = await control(slow, budget_s=0.05).screen_listing("web", [CLEAN], snapshot)
    assert hidden == {"fetch"}
    assert asyncio.get_running_loop().time() - started < 0.25


@DENIED
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
    [
        pytest.param(*case[1:], marks=ALLOWED if case[3] is Decision.ALLOW else DENIED)
        for case in CALL_CASES
    ],
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


@DENIED
async def test_a_call_without_a_verified_listing_fails_closed(call):
    verdict = await control().evaluate(call("fetch"), Stage.PRE, BLOCK)
    assert (verdict.decision, verdict.reason_code) == (
        Decision.BLOCK,
        "tool_definition_unavailable",
    )


@DENIED
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


@DENIED
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


@DENIED
async def test_a_duplicate_past_the_listing_cap_hides_its_name(snapshot):
    """Codex P2 #6: 256 benign ``notes`` hid the unclassified, poisoned 257th."""
    tools = [tool("notes", "Saves a note.") for _ in range(MAX_LISTED_TOOLS)]
    tools.append(tool("notes", f"Saves a note. {INJECT_MARKER}"))
    assert await control().screen_listing("web", tools, snapshot) == {"notes"}


# ------------------------------------------------------------ judge band (2026-10-04)

POISONED = InjectionJudgement(is_injection=True, confidence=0.9, rationale="asks for secrets")
CLEARED = InjectionJudgement(is_injection=False, confidence=0.9, rationale="usage notes only")
DOUBTFUL = tool("fetch", f"Fetches pages. {DOUBT_MARKER}")


def judged(judge: Any, *, budget_s: float = 10.0, entries: int = 4096) -> ToolPoisoningControl:
    return ToolPoisoningControl(
        ClassifierRunner(MarkerClassifier()),
        judge,
        listing_budget_s=budget_s,
        judge_cache_entries=entries,
    )


BAND_CASES = [
    pytest.param(ScriptedJudge(POISONED), {"fetch"}, "tool_poisoning_detected", id="judge-poisoned",
                 marks=DENIED),
    pytest.param(ScriptedJudge(CLEARED), set(), "judge_cleared", id="judge-cleared", marks=ALLOWED),
    pytest.param(ScriptedJudge(None, JudgeResult.TIMEOUT), {"fetch"}, "judge_unavailable",
                 id="judge-timeout", marks=DENIED),
    pytest.param(ScriptedJudge(None, JudgeResult.SCHEMA_MISMATCH), {"fetch"}, "judge_unavailable",
                 id="judge-garbled", marks=DENIED),
    pytest.param(None, {"fetch"}, "judge_unavailable", id="no-judge", marks=DENIED),
]  # fmt: skip


@pytest.mark.parametrize(("judge", "hidden", "reason_code"), BAND_CASES)
async def test_the_judge_band_decides_uncertain_definitions(
    snapshot, call, judge, hidden, reason_code
):
    """Real-model gap: poisoned descriptions scoring 0.60 and 0.76 passed with no judge tier."""
    assert await judged(judge).screen_listing("web", [DOUBTFUL], snapshot) == hidden
    with listing_scope(listing(DOUBTFUL)):
        verdict = await judged(judge).evaluate(call("fetch"), Stage.PRE, BLOCK)
    assert verdict.reason_code == reason_code
    assert verdict.decision is (Decision.BLOCK if hidden else Decision.ALLOW)
    if judge is not None:
        assert DOUBT_MARKER in judge.contents[0]  # the definition text is what is judged


async def test_each_definition_is_judged_once_per_process(snapshot):
    judge = ScriptedJudge(CLEARED)
    screen = judged(judge)
    for _ in range(3):
        assert await screen.screen_listing("web", [DOUBTFUL, DOUBTFUL], snapshot) == set()
    assert len(judge.contents) == 1


async def test_an_unavailable_answer_is_not_remembered(snapshot):
    judge = ScriptedJudge(None)
    screen = judged(judge)
    assert await screen.screen_listing("web", [DOUBTFUL], snapshot) == {"fetch"}
    judge.answer = CLEARED
    assert await screen.screen_listing("web", [DOUBTFUL], snapshot) == set()
    assert len(judge.contents) == 2


class ContentJudge:
    """Poisoned when the definition contains ``EVIL``."""

    def __init__(self) -> None:
        self.contents: list[str] = []

    async def judge(self, *, control_id, instructions, content, response_model):
        del control_id, instructions
        self.contents.append(content)
        return response_model.model_validate(
            {"is_injection": "EVIL" in content, "confidence": 0.9, "rationale": "r"}
        )


async def test_a_poisoned_answer_survives_cache_eviction(snapshot):
    """Answers stored while deciding (cache of 1) must not turn a poisoned verdict clean."""
    tools = [
        tool("a", f"First. {DOUBT_MARKER}"),
        tool("b", f"Second EVIL. {DOUBT_MARKER}"),
        tool("c", f"Third. {DOUBT_MARKER}"),
    ]
    judge = ContentJudge()
    assert await judged(judge, entries=1).screen_listing("web", tools, snapshot) == {"b"}
    assert len(judge.contents) == 3


class SlowJudge:
    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s
        self.calls = 0

    async def judge(self, *, control_id, instructions, content, response_model):
        del control_id, instructions, content
        self.calls += 1
        await asyncio.sleep(self.delay_s)
        return response_model.model_validate(
            {"is_injection": False, "confidence": 0.9, "rationale": "r"}
        )


@DENIED
async def test_a_judge_slower_than_the_listing_deadline_hides_the_tool(snapshot):
    judge = SlowJudge(0.3)
    screen = judged(judge, budget_s=0.05)
    started = asyncio.get_running_loop().time()
    assert await screen.screen_listing("web", [DOUBTFUL], snapshot) == {"fetch"}
    assert asyncio.get_running_loop().time() - started < 0.25
    await asyncio.sleep(0.4)  # the judge call kept running and its answer was remembered
    assert await screen.screen_listing("web", [DOUBTFUL], snapshot) == set()
    assert judge.calls == 1


def test_judge_band_without_judges_is_a_policy_error(policy_doc, snapshot_from):
    policy_doc.pop("judges", None)
    policy_doc["controls"]["tool_poisoning"] = {"judge_band": [0.5, 0.85]}
    with pytest.raises(PolicyLoadError, match=r"tool_poisoning\.judge_band"):
        snapshot_from(policy_doc)


def test_judge_band_must_be_ordered():
    with pytest.raises(ValueError, match="must be ordered"):
        ToolPoisoningConfig(judge_band=(0.9, 0.5))


class DilutedClassifier:
    """Fires on the marker only in short texts, as the real model is diluted by a long
    definition (a poisoned description: 0.76 alone, 0.01 inside the whole definition)."""

    def __call__(self, texts: Sequence[str]) -> list[InjectionScore]:
        return [
            InjectionScore(0.99 if INJECT_MARKER in t and len(t) < 120 else 0.0, 0, len(t))
            for t in texts
        ]


@DENIED
async def test_a_poisoned_field_is_classified_on_its_own(snapshot, call):
    padding = {f"option_{i}": {"type": "string", "title": f"Option {i}"} for i in range(12)}
    definition = tool("fetch", f"Fetches pages. {INJECT_MARKER}", **padding)
    assert len(definition_text(definition)) > 120
    screen = ToolPoisoningControl(ClassifierRunner(DilutedClassifier()))
    assert await screen.screen_listing("web", [definition], snapshot) == {"fetch"}
    with listing_scope(listing(definition)):
        verdict = await screen.evaluate(call("fetch"), Stage.PRE, BLOCK)
    assert verdict.reason_code == "tool_poisoning_detected"


# ------------------------------------------- judge cache vs policy reload (2026-10-04)


class ModelUpstream(Upstream):
    """A scripted LLM upstream for the real `JudgeClient`: the answer depends on the model
    the judge asks, so a reused answer from another model is visible."""

    def __init__(self, poisoned_by: dict[str, bool], delay_s: float = 0.0) -> None:
        self.poisoned_by = poisoned_by
        self.delay_s = delay_s
        self.models: list[str] = []

    async def execute(self, payload: object, snapshot: Any) -> UpstreamResult:
        del snapshot
        model = cast("dict[str, Any]", payload)["model"]
        self.models.append(model)
        await asyncio.sleep(self.delay_s)
        verdict = {"is_injection": self.poisoned_by[model], "confidence": 0.9, "rationale": "r"}
        body = {"choices": [{"message": {"role": "assistant", "content": json.dumps(verdict)}}]}
        return UpstreamResult(body=body, elapsed_s=0.0)


def with_judge_model(policy_doc: dict[str, Any], snapshot_from: Any, model: str) -> Any:
    document = copy.deepcopy(policy_doc)
    set_judges(document, {"model": model, "timeout_s": 5})
    return snapshot_from(document)


@DENIED
async def test_a_reload_to_another_judge_model_asks_again(policy_doc, snapshot_from):
    """Codex P2: a clearance from the old judge model was reused after a reload."""
    old = with_judge_model(policy_doc, snapshot_from, "lenient-judge")
    new = with_judge_model(policy_doc, snapshot_from, "strict-judge")
    current = {"snapshot": old}
    upstream = ModelUpstream({"lenient-judge": False, "strict-judge": True})
    client = JudgeClient(upstream, lambda: current["snapshot"])
    screen = ToolPoisoningControl(ClassifierRunner(MarkerClassifier()), client)
    assert await screen.screen_listing("web", [DOUBTFUL], old) == set()  # old model: clean
    current["snapshot"] = new
    assert await screen.screen_listing("web", [DOUBTFUL], new) == {"fetch"}  # asked again
    assert upstream.models == ["lenient-judge", "strict-judge"]
    assert await screen.screen_listing("web", [DOUBTFUL], new) == {"fetch"}
    assert upstream.models == ["lenient-judge", "strict-judge"]  # cached per configuration


async def test_a_judge_call_started_before_a_reload_keeps_its_policy(policy_doc, snapshot_from):
    """The listing timed out under the old policy with judge calls still queued (more than
    `MAX_CONCURRENT_JUDGES`); they run after the reload, under the old model, and fill only
    the old configuration's entries."""
    old = with_judge_model(policy_doc, snapshot_from, "lenient-judge")
    new = with_judge_model(policy_doc, snapshot_from, "strict-judge")
    current = {"snapshot": old}
    upstream = ModelUpstream({"lenient-judge": False, "strict-judge": True}, delay_s=0.2)
    client = JudgeClient(upstream, lambda: current["snapshot"])
    screen = ToolPoisoningControl(
        ClassifierRunner(MarkerClassifier()), client, listing_budget_s=0.05
    )
    tools = [tool(f"t{i}", f"Tool {i}. {DOUBT_MARKER}") for i in range(MAX_CONCURRENT_JUDGES + 1)]
    hidden = await screen.screen_listing("web", tools, old)
    assert hidden == {t.name for t in tools}  # no answer in time
    current["snapshot"] = new  # reload while calls are in flight or still queued
    await asyncio.sleep(0.6)
    assert upstream.models == ["lenient-judge"] * len(tools)  # each under its own policy
    patient = ToolPoisoningControl(ClassifierRunner(MarkerClassifier()), client)
    patient._answers = screen._answers  # the same cache, without the short deadline
    assert await patient.screen_listing("web", tools, new) == {t.name for t in tools}
    assert upstream.models.count("strict-judge") == len(tools)  # the new model was asked
