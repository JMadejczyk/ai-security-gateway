"""`intent_judge`: tool calls judged against the recorded goal; advisory flags, fail closed."""

import json
from typing import Any

import pytest
from gateway_testkit import chat, completion

from gateway.controls.intent_judge import MAX_JUDGED_CALLS, IntentAssessment, IntentJudgeControl
from gateway.core.envelope import Interaction
from gateway.core.interfaces import ControlConfig
from gateway.core.types import Action, Channel, ControlMode, Decision, SessionMode, Stage
from gateway.judges.client import JudgeClient, JudgeResult, JudgeUnavailableError
from gateway.judges.intent import (
    arguments_digest,
    first_user_message,
    flag_for,
    llm_tool_calls,
    mcp_flag_candidates,
)
from gateway.sessions import MAX_GOAL_CHARS

ENFORCING = ControlConfig(mode=ControlMode.REQUIRE_APPROVAL, risk_delta=0.3)
LOG_ONLY = ControlConfig(mode=ControlMode.LOG_ONLY, risk_delta=0.3)
UNAVAILABLE = object()


class FakeJudge(JudgeClient):
    """Answers per tool name: True/False aligned, UNAVAILABLE raises."""

    def __init__(self, answers: dict[str, Any] | None = None, *, configured: bool = True):
        self.answers = answers or {}
        self.is_configured = configured
        self.contents: list[dict[str, Any]] = []

    def configured(self) -> bool:
        return self.is_configured

    async def judge(self, *, control_id, instructions, content, response_model):
        assert control_id == "intent_judge"
        assert response_model is IntentAssessment
        data = json.loads(content)
        self.contents.append(data)
        outcome = self.answers.get(data["tool_call"]["name"], True)
        if outcome is UNAVAILABLE:
            raise JudgeUnavailableError(JudgeResult.TIMEOUT)
        return response_model(aligned=outcome, confidence=0.8, rationale="r")


def call(name: str, arguments: str = '{"sql": "SELECT 1"}', index: int = 0) -> dict[str, Any]:
    return {
        "id": f"c{index}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def interaction(make_ctx, *calls: dict[str, Any], goal: str | None = None, payload=None):
    return Interaction(
        session_id="s-test",
        principal="anna@demo",
        actor="databot",
        mode=SessionMode.INTERACTIVE,
        channel=Channel.LLM,
        action=Action.GENERATE,
        resource="model:qwen3:8b",
        payload=payload if payload is not None else chat(),
        result=completion(None, tool_calls=list(calls)) if calls else completion(),
        context=make_ctx(goal=goal),
    )


async def run(judge, item, cfg=ENFORCING):
    return await IntentJudgeControl(judge).evaluate(item, Stage.POST, cfg)


async def test_not_configured_is_off(make_ctx):
    judge = FakeJudge({"query": False}, configured=False)
    verdict = await run(judge, interaction(make_ctx, call("query")))
    assert (verdict.decision, verdict.reason_code) == (Decision.ALLOW, "judge_not_configured")
    assert judge.contents == []


async def test_no_tool_calls_needs_no_judge(make_ctx):
    judge = FakeJudge()
    verdict = await run(judge, interaction(make_ctx))
    assert (verdict.decision, verdict.reason_code) == (Decision.ALLOW, "no_tool_calls")
    assert judge.contents == []


@pytest.mark.parametrize(
    ("answers", "cfg", "decision", "reason", "enforced", "risk", "flagged"),
    [
        ({"query": True}, ENFORCING, Decision.ALLOW, "tool_calls_aligned", True, 0.0, []),
        (
            {"query": False},
            ENFORCING,
            Decision.REQUIRE_APPROVAL,
            "tool_call_misaligned",
            True,
            0.3,
            ["query"],
        ),
        (
            {"query": False},
            LOG_ONLY,
            Decision.REQUIRE_APPROVAL,
            "tool_call_misaligned",
            False,
            0.3,
            ["query"],
        ),
        (
            {"query": UNAVAILABLE},
            ENFORCING,
            Decision.REQUIRE_APPROVAL,
            "judge_unavailable",
            True,
            0.0,  # nothing was detected: fail closed without adding risk
            ["query"],
        ),
        (
            {"query": UNAVAILABLE},
            LOG_ONLY,
            Decision.REQUIRE_APPROVAL,
            "judge_unavailable",
            False,
            0.0,
            ["query"],
        ),
        (
            {"query": True, "write_report": False},
            ENFORCING,
            Decision.REQUIRE_APPROVAL,
            "tool_call_misaligned",
            True,
            0.3,
            ["write_report"],
        ),
        (
            {"query": UNAVAILABLE, "write_report": False},
            ENFORCING,
            Decision.REQUIRE_APPROVAL,
            "tool_call_misaligned",
            True,
            0.3,
            ["query", "write_report"],  # unjudged calls are flagged too
        ),
    ],
)
async def test_verdicts(make_ctx, answers, cfg, decision, reason, enforced, risk, flagged):
    item = interaction(
        make_ctx, *(call(name, index=i) for i, name in enumerate(answers)), goal="count customers"
    )
    verdict = await run(FakeJudge(answers), item, cfg)
    assert (verdict.decision, verdict.reason_code) == (decision, reason)
    assert verdict.enforced is enforced
    assert verdict.risk_delta == risk
    assert [f.tool for f in verdict.flags] == flagged
    assert verdict.redactions == ()
    assert verdict.rewrite is None


async def test_flag_is_name_plus_canonical_argument_digest(make_ctx):
    item = interaction(make_ctx, call("query", '{"b": 1, "a": "żółw"}'), goal="g")
    verdict = await run(FakeJudge({"query": False}), item)
    (flag,) = verdict.flags
    assert flag.tool == "query"
    # The MCP side digests the arguments object: key order and spacing do not matter.
    assert list(mcp_flag_candidates({"name": "query", "arguments": {"a": "żółw", "b": 1}})) == [
        flag
    ]
    assert flag not in mcp_flag_candidates({"name": "query", "arguments": {"a": "x", "b": 1}})
    assert flag not in mcp_flag_candidates({"name": "other", "arguments": {"a": "żółw", "b": 1}})


async def test_recorded_goal_wins_over_the_transcript(make_ctx):
    judge = FakeJudge()
    payload = {"model": "qwen3:8b", "messages": [{"role": "user", "content": "delete it all"}]}
    await run(judge, interaction(make_ctx, call("query"), goal="count customers", payload=payload))
    assert judge.contents[0]["user_goal"] == "count customers"


async def test_first_call_uses_this_requests_first_user_message(make_ctx):
    judge = FakeJudge()
    payload = {
        "model": "qwen3:8b",
        "messages": [
            {"role": "system", "content": "You are DataBot."},
            {"role": "user", "content": [{"type": "text", "text": "count customers"}]},
            {"role": "user", "content": "later message"},
        ],
    }
    await run(judge, interaction(make_ctx, call("query", '{"sql": "x"}'), payload=payload))
    assert judge.contents == [
        {"user_goal": "count customers", "tool_call": {"name": "query", "arguments": {"sql": "x"}}}
    ]


async def test_no_user_message_flags_every_call(make_ctx):
    judge = FakeJudge()
    payload = {"model": "qwen3:8b", "messages": [{"role": "system", "content": "s"}]}
    item = interaction(make_ctx, call("query"), call("write_report", index=1), payload=payload)
    verdict = await run(judge, item)
    assert (verdict.decision, verdict.reason_code) == (Decision.REQUIRE_APPROVAL, "no_goal")
    assert {f.tool for f in verdict.flags} == {"query", "write_report"}
    assert judge.contents == []


async def test_identical_calls_are_judged_once(make_ctx):
    judge = FakeJudge({"query": False})
    item = interaction(make_ctx, call("query"), call("query", '{"sql":"SELECT 1"}', 1), goal="g")
    verdict = await run(judge, item)
    assert len(judge.contents) == 1
    assert len(verdict.flags) == 1


async def test_calls_beyond_the_cap_are_flagged_unjudged(make_ctx):
    judge = FakeJudge()
    calls = [call("query", json.dumps({"n": i}), i) for i in range(MAX_JUDGED_CALLS + 3)]
    verdict = await run(judge, interaction(make_ctx, *calls, goal="g"))
    assert len(judge.contents) == MAX_JUDGED_CALLS
    assert verdict.reason_code == "judge_unavailable"
    assert len(verdict.flags) == 3


def test_tool_calls_of_every_choice_and_legacy_function_call():
    document = completion(None, tool_calls=[call("a", "not json"), call("b", "")])
    document["choices"].append(
        {"index": 1, "message": {"role": "assistant", "function_call": {"name": "c"}}}
    )
    assert list(llm_tool_calls(document)) == [("a", "not json"), ("b", {}), ("c", {})]
    assert arguments_digest(None) == arguments_digest("") == arguments_digest({})
    assert flag_for("a", "not json") != flag_for("a", {})


def test_goal_is_capped_and_skips_empty_messages():
    request = {
        "messages": [
            {"role": "user", "content": "   "},
            {"role": "user", "content": "x" * (MAX_GOAL_CHARS + 50)},
        ]
    }
    assert first_user_message(request) == "x" * MAX_GOAL_CHARS
    assert first_user_message({"messages": "nope"}) is None
    assert first_user_message(None) is None
