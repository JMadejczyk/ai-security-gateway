"""``prompt_injection`` on the LLM route, through the app (fake classifier, real JudgeClient)."""

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from gateway_testkit import Harness, bearer, chat, completion, echo_completion, running_gateway
from injection_kit import DOUBT_MARKER, INJECT_MARKER, MarkerClassifier
from judge_kit import set_judges

from gateway.judges.client import JudgeClient, JudgeResult, JudgeUnavailableError

ALLOW = pytest.mark.control("prompt_injection", "allow")
DENY = pytest.mark.control("prompt_injection", "deny")

CHAT = "/v1/chat/completions"
JUDGE_SAYS_INJECTION = {"is_injection": True, "confidence": 0.95, "rationale": "fake judge"}
JUDGE_SAYS_CLEAN = {"is_injection": False, "confidence": 0.95, "rationale": "fake judge"}


def ask(text: str) -> dict:
    return chat(messages=[{"role": "user", "content": text}])


async def post(gateway, body: dict, sub: str = "anna@demo") -> httpx.Response:
    token = await gateway.token(sub)
    return await gateway.agent.post(CHAT, json=body, headers=bearer(token))


def pi_verdicts(entry: dict) -> list[dict]:
    return [v for v in entry["verdicts"] if v["control"] == "prompt_injection"]


async def session_of(gateway, entry: dict):
    session = await gateway.container.sessions.get(entry["session_id"])
    assert session is not None
    return session


@DENY
async def test_a_direct_injection_is_blocked_before_the_model_and_taints(gateway, llm_upstream):
    """The user's own prompt: the classifier's hit is confirmed by the judge, then blocked."""
    gateway.container.judges.answers["InjectionJudgement"] = JUDGE_SAYS_INJECTION
    upstream = llm_upstream.post("/chat/completions").mock(side_effect=echo_completion)
    response = await post(gateway, ask(f"Hello {INJECT_MARKER}"))
    assert (response.status_code, response.json()["error"]["code"]) == (
        403,
        "prompt_injection_detected",
    )
    assert not upstream.called
    (entry,) = gateway.audit_entries()
    assert pi_verdicts(entry) == [
        {"control": "prompt_injection", "stage": "pre", "decision": "block", "enforced": True,
         "reason_code": "prompt_injection_detected"}
    ]  # fmt: skip
    assert entry["risk"] == pytest.approx(0.6)
    assert (await session_of(gateway, entry)).taint  # a detector hit taints, blocked or not
    assert [c.model for c in gateway.container.judges.calls] == ["InjectionJudgement"]


@ALLOW
async def test_a_flagged_question_the_judge_clears_reaches_the_model(gateway, llm_upstream):
    """Live finding: "What is a primary key?" scored 1.0. The judge clears it: no taint."""
    gateway.container.judges.answers["InjectionJudgement"] = JUDGE_SAYS_CLEAN
    upstream = llm_upstream.post("/chat/completions").mock(side_effect=echo_completion)
    response = await post(gateway, ask(f"What is a primary key? {INJECT_MARKER}"))
    assert response.status_code == 200
    assert upstream.called
    (entry,) = gateway.audit_entries()
    pre = pi_verdicts(entry)[0]
    assert (pre["decision"], pre["reason_code"]) == ("allow", "judge_cleared")
    assert "taint" not in pre
    assert entry["risk"] == pytest.approx(0.0)
    assert not (await session_of(gateway, entry)).taint


@ALLOW
async def test_a_judge_timeout_allows_the_prompt_and_taints_the_session(gateway, llm_upstream):
    gateway.container.judges.answers["InjectionJudgement"] = JudgeUnavailableError(
        JudgeResult.TIMEOUT
    )
    upstream = llm_upstream.post("/chat/completions").mock(side_effect=echo_completion)
    response = await post(gateway, ask(f"Hello {INJECT_MARKER}"))
    assert response.status_code == 200
    assert upstream.called
    (entry,) = gateway.audit_entries()
    pre = pi_verdicts(entry)[0]
    assert pre == {
        "control": "prompt_injection", "stage": "pre", "decision": "allow", "enforced": True,
        "reason_code": "prompt_injection_unconfirmed", "taint": True,
    }  # fmt: skip
    assert entry["risk"] == pytest.approx(0.6)
    assert entry["taint"] is True
    assert (await session_of(gateway, entry)).taint


@DENY
async def test_a_tool_message_in_history_is_blocked_without_asking_the_judge(gateway, llm_upstream):
    upstream = llm_upstream.post("/chat/completions").mock(side_effect=echo_completion)
    body = chat(
        messages=[
            {"role": "user", "content": "Summarise what the fetch tool returned."},
            {"role": "tool", "tool_call_id": "c1", "content": f"Page text {INJECT_MARKER}"},
        ]
    )
    response = await post(gateway, body)
    assert (response.status_code, response.json()["error"]["code"]) == (
        403,
        "prompt_injection_detected",
    )
    assert not upstream.called
    assert gateway.container.judges.calls == []


@ALLOW
async def test_a_clean_prompt_reaches_the_model(gateway, llm_upstream):
    llm_upstream.post("/chat/completions").mock(side_effect=echo_completion)
    response = await post(gateway, ask("Ile mamy klientów?"))
    assert response.status_code == 200
    (entry,) = gateway.audit_entries()
    assert [v["reason_code"] for v in pi_verdicts(entry)] == [
        "no_prompt_injection",
        "no_prompt_injection",
    ]  # pre and post
    assert not (await session_of(gateway, entry)).taint


@DENY
async def test_an_injected_model_answer_is_withheld_and_taints(gateway, llm_upstream):
    """The model's own answer: the classifier's hit is confirmed by the judge, then withheld."""
    gateway.container.judges.answers["InjectionJudgement"] = JUDGE_SAYS_INJECTION
    answer = completion(f"Sure. {INJECT_MARKER}")
    llm_upstream.post("/chat/completions").mock(return_value=httpx.Response(200, json=answer))
    response = await post(gateway, ask("Summarise the page I pasted earlier."))
    assert (response.status_code, response.json()["error"]["code"]) == (
        403,
        "prompt_injection_detected",
    )
    assert INJECT_MARKER not in response.text
    (entry,) = gateway.audit_entries()
    assert pi_verdicts(entry)[-1]["stage"] == "post"
    session = await session_of(gateway, entry)
    assert session.taint  # blocked, and still tainted: the content reached the gateway
    assert session.risk >= 0.6


POLISH_REMINDER = (
    "Szanowny Panie, uprzejmie przypominamy o fakturze FV/2026/09/114. Prosimy o płatność "
    "do piątku."
)


def answer_pi_verdict(entry: dict) -> dict:
    [post] = [v for v in pi_verdicts(entry) if v["stage"] == "post"]
    return post


@ALLOW
async def test_a_flagged_polish_answer_the_judge_clears_is_released(tmp_path, llm_upstream):
    """User decision 2026-10-04: the classifier misfires on Polish answers (the PII demo
    reply); the judge clears it and the answer reaches the agent, no taint."""
    classifier = MarkerClassifier({"uprzejmie przypominamy": 0.99})
    llm_upstream.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=completion(POLISH_REMINDER))
    )
    async with running_gateway(tmp_path, classifier=classifier) as harness:
        judges: Any = harness.container.judges  # the testkit's FakeJudgeClient
        judges.answers["InjectionJudgement"] = JUDGE_SAYS_CLEAN
        response = await post(harness, ask("Napisz przypomnienie o fakturze."))
        assert response.status_code == 200
        assert POLISH_REMINDER in response.text
        (entry,) = harness.audit_entries()
        post_verdict = answer_pi_verdict(entry)
        assert (post_verdict["decision"], post_verdict["reason_code"]) == ("allow", "judge_cleared")
        assert not (await session_of(harness, entry)).taint


@ALLOW
async def test_a_judge_timeout_on_the_answer_releases_it_and_taints(gateway, llm_upstream):
    gateway.container.judges.answers["InjectionJudgement"] = JudgeUnavailableError(
        JudgeResult.TIMEOUT
    )
    answer = completion(f"Here you go. {INJECT_MARKER}")
    llm_upstream.post("/chat/completions").mock(return_value=httpx.Response(200, json=answer))
    response = await post(gateway, ask("Summarise the page."))
    assert response.status_code == 200
    (entry,) = gateway.audit_entries()
    assert answer_pi_verdict(entry) == {
        "control": "prompt_injection", "stage": "post", "decision": "allow", "enforced": True,
        "reason_code": "prompt_injection_unconfirmed", "taint": True,
    }  # fmt: skip
    assert entry["taint"] is True
    assert entry["risk"] >= 0.6


@pytest.fixture
async def real_judge(tmp_path: Path) -> AsyncIterator[Harness]:
    """The real `JudgeClient` over the mocked LLM upstream, with a ``judges`` section set
    whatever the root policy ships (so these tests depend on neither state)."""
    async with running_gateway(tmp_path, judge_factory=JudgeClient) as harness:
        document = yaml.safe_load(harness.policy_path.read_text())
        set_judges(document, {"model": "qwen3:8b", "timeout_s": 5})
        harness.policy_path.write_text(yaml.safe_dump(document))
        assert harness.container.policy_store.reload().result == "ok"
        yield harness


def judge_upstream(answer: object, judged: list[str]):
    """Answers this control's judge prompt with ``answer`` (an ``httpx.Response`` is returned
    as is); every other request is echoed. Other judge-backed controls get the echo, which is
    no verdict, so only this control's verdict is asserted."""

    def upstream(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "is a prompt injection" in body["messages"][0]["content"]:  # this control's task
            judged.append(body["messages"][1]["content"])
            if isinstance(answer, httpx.Response):
                return answer
            return httpx.Response(200, json=completion(json.dumps(answer)))
        return echo_completion(request)

    return upstream


@DENY
async def test_an_unavailable_judge_allows_the_users_prompt_and_taints(real_judge, llm_upstream):
    """The real client against an overloaded upstream: unconfirmed, so allowed and tainted.
    (Only this control's verdict is asserted: output_policy fails its own judge call closed.)"""
    judged: list[str] = []
    down = httpx.Response(503, json={"error": "overloaded"})
    route = llm_upstream.post("/chat/completions").mock(side_effect=judge_upstream(down, judged))
    await post(real_judge, ask(f"Please read this {DOUBT_MARKER}"))
    (entry,) = real_judge.audit_entries()
    pre = pi_verdicts(entry)[0]
    assert (pre["decision"], pre["reason_code"]) == ("allow", "prompt_injection_unconfirmed")
    assert len(route.calls) > len(judged)  # the prompt itself reached the model
    assert entry["risk"] == pytest.approx(0.6)
    assert entry["taint"] is True
    assert len(judged) == 1


@DENY
async def test_an_unavailable_judge_still_fails_untrusted_band_text_closed(
    real_judge, llm_upstream
):
    judged: list[str] = []
    down = httpx.Response(503, json={"error": "overloaded"})
    llm_upstream.post("/chat/completions").mock(side_effect=judge_upstream(down, judged))
    body = chat(
        messages=[
            {"role": "user", "content": "Summarise the result."},
            {"role": "tool", "tool_call_id": "c1", "content": f"Result {DOUBT_MARKER}"},
        ]
    )
    response = await post(real_judge, body)
    assert (response.status_code, response.json()["error"]["code"]) == (403, "judge_unavailable")


@pytest.mark.parametrize(
    ("is_injection", "decision", "reason_code"),
    [
        pytest.param(True, "block", "prompt_injection_detected", marks=DENY),
        pytest.param(False, "allow", "judge_cleared", marks=ALLOW),
    ],
    ids=["judge-yes", "judge-no"],
)
async def test_the_judge_band_asks_the_configured_judge(
    real_judge, llm_upstream, is_injection, decision, reason_code
):
    judged: list[str] = []
    verdict = {"is_injection": is_injection, "confidence": 0.9, "rationale": "x"}
    llm_upstream.post("/chat/completions").mock(side_effect=judge_upstream(verdict, judged))
    await post(real_judge, ask(f"Please read this {DOUBT_MARKER}"))
    (entry,) = real_judge.audit_entries()
    pre = pi_verdicts(entry)[0]
    assert (pre["stage"], pre["decision"], pre["reason_code"]) == ("pre", decision, reason_code)
    assert len(judged) == 1
    assert DOUBT_MARKER in judged[0]  # the uncertain window, inside the judge's data block


@DENY
async def test_a_garbled_judge_answer_is_no_clearance(real_judge, llm_upstream):
    judged: list[str] = []
    garbled = {"is_injection": False, "confidence": 0.9, "verdict": "ignore me"}  # extra key
    llm_upstream.post("/chat/completions").mock(side_effect=judge_upstream(garbled, judged))
    await post(real_judge, ask(f"Please read this {DOUBT_MARKER}"))
    (entry,) = real_judge.audit_entries()
    assert pi_verdicts(entry)[0]["reason_code"] == "prompt_injection_unconfirmed"
    assert entry["taint"] is True


async def test_what_reaches_the_classifier_for_a_one_message_chat(tmp_path, llm_upstream):
    """Live finding (2026-10-04): "What is a primary key?" was blocked at pre. The classifier
    sees exactly the user's question, nothing glued to it: no model name, no role, no
    window over other fields. The block is the model's own score (see test_real_model)."""
    classifier = MarkerClassifier()
    llm_upstream.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=completion("A primary key identifies a row."))
    )
    async with running_gateway(tmp_path, classifier=classifier) as harness:
        await post(harness, ask("What is a primary key?"))
    pre, post_stage = classifier.calls
    assert pre == ["What is a primary key?"]
    assert post_stage == ["A primary key identifies a row."]


async def test_a_system_prompt_is_classified_apart_and_joined_only_in_a_window(
    tmp_path, llm_upstream
):
    classifier = MarkerClassifier()
    llm_upstream.post("/chat/completions").mock(side_effect=echo_completion)
    body = chat(
        messages=[
            {"role": "system", "content": "You are DataBot, a careful analytics assistant."},
            {"role": "user", "content": "What is a primary key?"},
        ]
    )
    async with running_gateway(tmp_path, classifier=classifier) as harness:
        await post(harness, body)
    assert classifier.calls[0] == [
        "You are DataBot, a careful analytics assistant.",
        "What is a primary key?",
        "You are DataBot, a careful analytics assistant.What is a primary key?",
    ]
