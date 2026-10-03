"""The root policy turns the judges on: the testkit's deterministic judge answers by default,
every judge-backed control uses it, and a test can override one answer."""

import json

import httpx
import pytest
from gateway_testkit import bearer, chat, completion
from injection_kit import DOUBT_MARKER
from judge_kit import FakeJudgeClient

from gateway.judges.client import JudgeResult, JudgeUnavailableError

CHAT = "/v1/chat/completions"
TOOL_CALL = {
    "id": "c1",
    "type": "function",
    "function": {"name": "query", "arguments": json.dumps({"sql": "SELECT 1"})},
}


def judge_of(gateway) -> FakeJudgeClient:
    judge = gateway.container.judges
    assert isinstance(judge, FakeJudgeClient)
    return judge


async def ask(gateway, text: str = "How many customers?") -> httpx.Response:
    token = await gateway.token("anna@demo")
    body = chat(messages=[{"role": "user", "content": text}])
    return await gateway.agent.post(CHAT, json=body, headers=bearer(token))


def verdicts(gateway, control: str) -> list[dict]:
    entry = gateway.audit_entries()[-1]
    return [v for v in entry["verdicts"] if v["control"] == control]


def test_root_policy_configures_the_judges(snapshot):
    judges = snapshot.policy.judges
    assert judges is not None
    assert judges.model == "qwen3:8b"
    assert snapshot.policy.resolved_control_mode("output_policy") == "redact"
    assert snapshot.policy.resolved_control_mode("intent_judge") == "require_approval"


async def test_default_judge_clears_every_judge_control(gateway, llm_upstream):
    llm_upstream.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=completion(None, tool_calls=[TOOL_CALL]))
    )
    response = await ask(gateway, f"Count customers {DOUBT_MARKER}")
    assert response.status_code == 200, response.text
    models = {call.model for call in judge_of(gateway).calls}
    assert models == {"IntentAssessment", "OutputAssessment", "InjectionJudgement"}
    # prompt_injection's judge band asked the judge instead of failing closed.
    (pre,) = [v for v in verdicts(gateway, "prompt_injection") if v["stage"] == "pre"]
    assert pre["reason_code"] == "judge_cleared"
    assert verdicts(gateway, "intent_judge")[0]["reason_code"] == "tool_calls_aligned"
    assert verdicts(gateway, "output_policy")[0]["reason_code"] == "answer_in_scope"
    assert len(llm_upstream.calls) == 1  # judges never reach the (mocked) upstream


async def test_one_answer_can_be_overridden_per_test(gateway, llm_upstream):
    llm_upstream.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=completion("Olga's card is 4111."))
    )
    judge_of(gateway).answers["OutputAssessment"] = {
        "violations": [{"quote": "4111", "reason": "payments table"}]
    }
    response = await ask(gateway)
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == (
        "Olga's card is [REDACTED:OUT_OF_SCOPE]."
    )


@pytest.mark.parametrize(
    "answer",
    [
        {},
        {"error": "unable to assess"},
        {"violation": []},
        JudgeUnavailableError(JudgeResult.TIMEOUT),
    ],
    ids=["empty", "error-object", "misspelled", "timeout"],
)
async def test_no_verdict_fails_output_policy_closed(gateway, llm_upstream, answer):
    llm_upstream.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=completion("There are 40 customers."))
    )
    judge_of(gateway).answers["OutputAssessment"] = answer
    response = await ask(gateway)
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "judge_unavailable"
