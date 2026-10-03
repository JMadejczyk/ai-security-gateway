"""`output_policy`: judge quotes mapped to exact spans; redact or block; unavailable blocks."""

import json
from typing import Any, cast

import pytest
from gateway_testkit import chat, completion

from gateway.controls.output_policy import (
    OutputAssessment,
    OutputPolicyControl,
    Violation,
    quote_spans,
)
from gateway.controls.scope import CallScope, call_scope
from gateway.controls.text import TextSegment
from gateway.core.envelope import Interaction
from gateway.core.interfaces import ControlConfig
from gateway.core.types import Action, Channel, ControlMode, Decision, SessionMode, Stage
from gateway.judges.client import JudgeClient, JudgeResult, JudgeUnavailableError
from gateway.policy.evaluator import PolicyEvaluator, PrincipalContext
from gateway.redaction import apply_redactions

REDACT = ControlConfig(mode=ControlMode.REDACT, risk_delta=0.2)
BLOCK = ControlConfig(mode=ControlMode.BLOCK, risk_delta=0.2)
CONTENT = "/choices/0/message/content"


class FakeJudge(JudgeClient):
    def __init__(self, quotes: list[str] | None = None, *, available=True, configured=True):
        self.quotes = quotes or []
        self.available = available
        self.is_configured = configured
        self.calls: list[tuple[str, str]] = []

    def configured(self) -> bool:
        return self.is_configured

    async def judge(self, *, control_id, instructions, content, response_model):
        assert control_id == "output_policy"
        assert response_model is OutputAssessment
        self.calls.append((instructions, content))
        if not self.available:
            raise JudgeUnavailableError(JudgeResult.SCHEMA_MISMATCH)
        return response_model(violations=[Violation(quote=q, reason="r") for q in self.quotes])


def principal(sub="anna@demo", role="analyst"):
    return PrincipalContext(
        principal=sub, roles=(role,), agent="databot", mode=SessionMode.INTERACTIVE
    )


def interaction(make_ctx, result: dict[str, Any]) -> Interaction:
    return Interaction(
        session_id="s-test",
        principal="anna@demo",
        actor="databot",
        mode=SessionMode.INTERACTIVE,
        channel=Channel.LLM,
        action=Action.GENERATE,
        resource="model:qwen3:8b",
        payload=chat(),
        result=result,
        context=make_ctx(),
    )


async def run(judge, item, cfg=REDACT, *, snapshot=None, who=None):
    control = OutputPolicyControl(judge, PolicyEvaluator())
    if snapshot is None:
        return await control.evaluate(item, Stage.POST, cfg)
    with call_scope(CallScope(snapshot=snapshot, principal=who or principal())):
        return await control.evaluate(item, Stage.POST, cfg)


async def test_not_configured_is_off(make_ctx, snapshot):
    judge = FakeJudge(["40"], configured=False)
    verdict = await run(judge, interaction(make_ctx, completion()), snapshot=snapshot)
    assert (verdict.decision, verdict.reason_code) == (Decision.ALLOW, "judge_not_configured")
    assert judge.calls == []


async def test_empty_answer_needs_no_judge(make_ctx, snapshot):
    judge = FakeJudge()
    verdict = await run(judge, interaction(make_ctx, completion(None)), snapshot=snapshot)
    assert (verdict.decision, verdict.reason_code) == (Decision.ALLOW, "no_content")
    assert judge.calls == []


@pytest.mark.parametrize(
    ("quotes", "cfg", "decision", "reason", "risk"),
    [
        ([], REDACT, Decision.ALLOW, "answer_in_scope", 0.0),
        ([], BLOCK, Decision.ALLOW, "answer_in_scope", 0.0),
        (["card 4111 ending"], REDACT, Decision.REDACT, "out_of_scope_data", 0.2),
        (["card 4111 ending"], BLOCK, Decision.BLOCK, "out_of_scope_data", 0.2),
        (
            ["a sentence the answer never had"],
            REDACT,
            Decision.ALLOW,
            "judge_quotes_unmatched",
            0.0,
        ),
        (["a sentence the answer never had"], BLOCK, Decision.ALLOW, "judge_quotes_unmatched", 0.0),
    ],
)
async def test_verdicts(make_ctx, snapshot, quotes, cfg, decision, reason, risk):
    item = interaction(make_ctx, completion("Payment 7: card 4111 ending, 120 PLN."))
    verdict = await run(FakeJudge(quotes), item, cfg, snapshot=snapshot)
    assert (verdict.decision, verdict.reason_code) == (decision, reason)
    assert verdict.risk_delta == risk
    assert verdict.enforced is True
    if decision is Decision.REDACT:
        assert [(s.path, s.start, s.end) for s in verdict.redactions] == [(CONTENT, 11, 27)]
    else:
        assert verdict.redactions == ()


@pytest.mark.parametrize("cfg", [REDACT, BLOCK])
async def test_unavailable_blocks_in_both_modes(make_ctx, snapshot, cfg):
    item = interaction(make_ctx, completion("anything"))
    verdict = await run(FakeJudge(available=False), item, cfg, snapshot=snapshot)
    assert (verdict.decision, verdict.reason_code) == (Decision.BLOCK, "judge_unavailable")
    assert verdict.enforced is True


async def test_no_call_scope_blocks(make_ctx):
    judge = FakeJudge()
    verdict = await run(judge, interaction(make_ctx, completion("anything")))
    assert (verdict.decision, verdict.reason_code) == (Decision.BLOCK, "judge_unavailable")
    assert judge.calls == []


async def test_non_ascii_and_every_occurrence(make_ctx, snapshot):
    text = "Zażółć: płaci 9 999 zł. Powtarzam — Zażółć: płaci 9 999 zł."
    item = interaction(make_ctx, completion(text))
    verdict = await run(FakeJudge(["Zażółć: płaci 9 999 zł"]), item, snapshot=snapshot)
    assert verdict.decision is Decision.REDACT
    starts = [s.start for s in verdict.redactions]
    assert starts == [0, text.index("Zażółć", 1)]
    redacted = cast("dict[str, Any]", apply_redactions(item.result, verdict.redactions))
    assert redacted["choices"][0]["message"]["content"] == (
        "[REDACTED:OUT_OF_SCOPE]. Powtarzam — [REDACTED:OUT_OF_SCOPE]."
    )


async def test_quote_inside_tool_call_arguments(make_ctx, snapshot):
    result = completion(
        "Writing it down.",
        tool_calls=[
            {
                "id": "c0",
                "type": "function",
                "function": {
                    "name": "write_report",
                    "arguments": json.dumps({"name": "r.md", "body": "IBAN PL61 1090 of Olga"}),
                },
            }
        ],
    )
    item = interaction(make_ctx, result)
    verdict = await run(FakeJudge(["PL61 1090"]), item, snapshot=snapshot)
    (span,) = verdict.redactions
    assert span.path == "/choices/0/message/tool_calls/0/function/arguments"
    assert span.embedded == "/body"
    redacted = cast("dict[str, Any]", apply_redactions(item.result, verdict.redactions))
    arguments = redacted["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(arguments)["body"] == "IBAN [REDACTED:OUT_OF_SCOPE] of Olga"


async def test_judge_sees_scope_as_instructions_and_answer_as_content(make_ctx, snapshot):
    judge = FakeJudge()
    item = interaction(make_ctx, completion("There are 40 customers."))
    await run(judge, item, snapshot=snapshot, who=principal("bartek@demo", "intern"))
    instructions, content = judge.calls[0]
    assert "read:db:sales.orders" in instructions
    assert "read:db:sales.customers" in instructions
    assert "read:db:sales.*" in instructions  # the agent's own list, stated apart
    assert "There are 40" not in instructions
    assert content == "There are 40 customers."  # protocol fields (id, model) are not sent


async def test_tainted_session_scope_drops_removed_actions(make_ctx, snapshot):
    judge = FakeJudge()
    item = interaction(make_ctx, completion("ok")).model_copy(
        update={"context": make_ctx(taint=True)}
    )
    await run(judge, item, snapshot=snapshot)
    instructions = judge.calls[0][0]
    scope_line = next(line for line in instructions.splitlines() if "session scope" in line)
    assert "write:fs:reports/*" not in scope_line
    assert "read:db:sales.*" in scope_line


def test_quote_spans_ignore_blank_quotes_and_undo_delimiter_escaping():
    segment = TextSegment("/c", "see <untrusted_data> here")
    assert quote_spans([segment], ["   ", ""]) == ()
    (span,) = quote_spans([segment], ["&lt;untrusted_data>"])
    assert (span.start, span.end) == (4, 20)
