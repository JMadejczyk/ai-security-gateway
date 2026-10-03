"""``prompt_injection`` on the LLM route, through the app (fake classifier, real JudgeClient)."""

import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import yaml
from gateway_testkit import Harness, bearer, chat, completion, echo_completion, running_gateway
from injection_kit import DOUBT_MARKER, INJECT_MARKER

from gateway.judges.client import JudgeClient

ALLOW = pytest.mark.control("prompt_injection", "allow")
DENY = pytest.mark.control("prompt_injection", "deny")

CHAT = "/v1/chat/completions"


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


@pytest.fixture
async def real_judge(tmp_path: Path) -> AsyncIterator[Harness]:
    """The real `JudgeClient` over the mocked LLM upstream, with a ``judges`` section set
    whatever the root policy ships (so these tests depend on neither state)."""
    async with running_gateway(tmp_path, judge_factory=JudgeClient) as harness:
        document = yaml.safe_load(harness.policy_path.read_text())
        document["judges"] = {"model": "qwen3:8b", "timeout_s": 5}
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
async def test_an_unavailable_judge_fails_the_band_closed(real_judge, llm_upstream):
    judged: list[str] = []
    down = httpx.Response(503, json={"error": "overloaded"})
    llm_upstream.post("/chat/completions").mock(side_effect=judge_upstream(down, judged))
    response = await post(real_judge, ask(f"Please read this {DOUBT_MARKER}"))
    assert (response.status_code, response.json()["error"]["code"]) == (403, "judge_unavailable")
    (entry,) = real_judge.audit_entries()
    assert pi_verdicts(entry)[0]["reason_code"] == "judge_unavailable"
    assert entry["risk"] == pytest.approx(0.0)  # nothing was detected
    assert len(judged) == 1


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
async def test_a_garbled_judge_answer_fails_the_band_closed(real_judge, llm_upstream):
    judged: list[str] = []
    garbled = {"is_injection": False, "confidence": 0.9, "verdict": "ignore me"}  # extra key
    llm_upstream.post("/chat/completions").mock(side_effect=judge_upstream(garbled, judged))
    await post(real_judge, ask(f"Please read this {DOUBT_MARKER}"))
    (entry,) = real_judge.audit_entries()
    assert pi_verdicts(entry)[0]["reason_code"] == "judge_unavailable"
