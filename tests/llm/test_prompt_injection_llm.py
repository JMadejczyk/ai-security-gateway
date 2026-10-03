"""``prompt_injection`` on the LLM route, through the app (fake classifier, real JudgeClient)."""

import json

import httpx
import pytest
from gateway_testkit import bearer, chat, completion, echo_completion
from injection_kit import DOUBT_MARKER, INJECT_MARKER

CHAT = "/v1/chat/completions"
JUDGES = 'judges: { model: "qwen3:8b", timeout_s: 5 }\n'


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


async def test_the_judge_band_fails_closed_without_a_judge(gateway, llm_upstream):
    """policy.yaml ships without a ``judges`` section: doubt cannot be resolved, so it blocks."""
    upstream = llm_upstream.post("/chat/completions").mock(side_effect=echo_completion)
    response = await post(gateway, ask(f"Please read this {DOUBT_MARKER}"))
    assert (response.status_code, response.json()["error"]["code"]) == (403, "judge_unavailable")
    assert not upstream.called
    (entry,) = gateway.audit_entries()
    assert entry["risk"] == pytest.approx(0.0)  # nothing was detected


@pytest.mark.parametrize(
    ("is_injection", "decision", "reason_code"),
    [(True, "block", "prompt_injection_detected"), (False, "allow", "judge_cleared")],
    ids=["judge-yes", "judge-no"],
)
async def test_the_judge_band_asks_the_configured_judge(
    gateway, llm_upstream, is_injection, decision, reason_code
):
    """Only this control's verdict is asserted: with ``judges`` on, the other judge-backed
    controls (``output_policy``, ``intent_judge``) ask this mock too and get no answer."""
    gateway.policy_path.write_text(gateway.policy_path.read_text() + JUDGES)
    assert gateway.container.policy_store.reload().result == "ok"
    judged: list[str] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "is a prompt injection" in body["messages"][0]["content"]:  # this control's task
            judged.append(body["messages"][1]["content"])
            verdict = {"is_injection": is_injection, "confidence": 0.9, "rationale": "x"}
            return httpx.Response(200, json=completion(json.dumps(verdict)))
        return echo_completion(request)

    llm_upstream.post("/chat/completions").mock(side_effect=upstream)
    await post(gateway, ask(f"Please read this {DOUBT_MARKER}"))
    (entry,) = gateway.audit_entries()
    pre = pi_verdicts(entry)[0]
    assert (pre["stage"], pre["decision"], pre["reason_code"]) == ("pre", decision, reason_code)
    assert len(judged) == 1
    assert DOUBT_MARKER in judged[0]  # the uncertain window, inside the judge's data block
