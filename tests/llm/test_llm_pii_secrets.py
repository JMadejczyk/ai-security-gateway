"""``pii`` and ``secrets`` on ``/v1/chat/completions`` through the real app (demo step 5).

A PESEL is masked before the prompt leaves the gateway (JSON and SSE alike), an API key in a
prompt is refused before the upstream is called, credentials and personal data in the
model's answer are blocked or masked on the way back, and no audit line ever carries them.
Under the permissive profile ``pii`` only records, while the mandatory ``secrets`` enforces.
"""

import json
from typing import Any

import httpx
import pytest
import respx
import yaml
from gateway_testkit import Harness, bearer, chat, completion

CHAT = "/v1/chat/completions"
PESEL = "44051401359"
API_KEY = "sk-proj-" + "Ab3De5Fg7Hi9Jk1Lm3No5Pq7Rs9Tu1Vw3Xy5Za7Bc9De"  # fake, assembled
STRIPE = "sk_live_" + "4eC39HqLyjWDarjtT1zdp7dc"


def edit_policy(gateway: Harness, *, profile: str | None = None, **controls: Any) -> None:
    """Replace the given ``controls:`` entries (and the profile), then reload."""
    document = yaml.safe_load(gateway.policy_path.read_text())
    if profile is not None:
        document["profile"] = profile
    document["controls"].update(controls)
    gateway.policy_path.write_text(yaml.safe_dump(document, sort_keys=False))
    assert gateway.container.policy_store.reload().error is None


async def ask(gateway: Harness, content: str, **body: Any) -> httpx.Response:
    token = await gateway.token("anna@demo")
    request = chat(messages=[{"role": "user", "content": content}], **body)
    return await gateway.agent.post(CHAT, json=request, headers=bearer(token))


def sent_content(route: Any) -> str:
    return json.loads(route.calls.last.request.content)["messages"][0]["content"]


def verdict(entry: dict[str, Any], control: str, stage: str) -> dict[str, Any]:
    (found,) = [v for v in entry["verdicts"] if (v["control"], v["stage"]) == (control, stage)]
    return found


def assert_nothing_leaked(gateway: Harness, *values: str) -> None:
    raw = gateway.audit.getvalue()
    for value in values:
        assert value not in raw


class ScriptedAnswer:
    """The mocked upstream answers with whatever ``content`` a test sets."""

    def __init__(self, router: respx.MockRouter) -> None:
        self.content = "There are 40 customers."
        self.route = router.post("/chat/completions").mock(side_effect=self._respond)

    def _respond(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion(self.content))


@pytest.fixture
def answer(llm_upstream) -> ScriptedAnswer:
    return ScriptedAnswer(llm_upstream)


@pytest.mark.parametrize("stream", [False, True])
async def test_pesel_in_the_prompt_is_redacted_before_the_upstream(gateway, answer, stream):
    response = await ask(gateway, f"Klient o numerze PESEL {PESEL} pyta o fakturę.", stream=stream)
    assert response.status_code == 200
    assert (
        sent_content(answer.route) == "Klient o numerze PESEL [REDACTED:PL_PESEL] pyta o fakturę."
    )
    assert ("data: [DONE]" in response.text) is stream
    (entry,) = gateway.audit_entries()
    assert entry["decision"] == "redact"
    assert verdict(entry, "pii", "pre") == {
        "control": "pii",
        "stage": "pre",
        "decision": "redact",
        "enforced": True,
        "reason_code": "pii_detected",
    }
    assert entry["risk"] == pytest.approx(0.1)
    assert_nothing_leaked(gateway, PESEL)


async def test_api_key_in_the_prompt_is_blocked_and_never_sent(gateway, answer):
    response = await ask(gateway, f"Use this key: {API_KEY} to call the API")
    assert (response.status_code, response.json()["error"]["code"]) == (403, "secret_detected")
    assert not answer.route.called
    (entry,) = gateway.audit_entries()
    assert (entry["decision"], entry["reason_code"]) == ("block", "secret_detected")
    assert entry["risk"] == pytest.approx(0.3)
    assert API_KEY not in response.text
    assert_nothing_leaked(gateway, API_KEY)


@pytest.mark.parametrize("stream", [False, True])
async def test_secret_in_the_answer_is_withheld(gateway, answer, stream):
    answer.content = f"Sure, the production key is {STRIPE}."
    response = await ask(gateway, "What is the Stripe key?", stream=stream)
    assert (response.status_code, response.json()["error"]["code"]) == (403, "secret_detected")
    assert answer.route.call_count == 1
    assert STRIPE not in response.text
    (entry,) = gateway.audit_entries()
    assert verdict(entry, "secrets", "post")["decision"] == "block"
    assert_nothing_leaked(gateway, STRIPE)


async def test_secret_in_a_tool_call_argument_is_withheld(gateway, llm_upstream):
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "fetch", "arguments": json.dumps({"token": STRIPE})},
    }
    llm_upstream.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=completion(None, tool_calls=[call]))
    )
    response = await ask(gateway, "Call the API.")
    assert (response.status_code, response.json()["error"]["code"]) == (403, "secret_detected")
    assert STRIPE not in response.text


@pytest.mark.parametrize("stream", [False, True])
async def test_secrets_in_redact_mode_mask_the_answer(gateway, answer, stream):
    edit_policy(gateway, secrets={"mode": "redact"})
    answer.content = f"Use {STRIPE} for billing."
    response = await ask(gateway, "Which key?", stream=stream)
    assert response.status_code == 200
    assert STRIPE not in response.text
    if stream:
        assert "[REDACTED:API_KEY]" in response.text
    else:
        assert response.json()["choices"][0]["message"]["content"] == (
            "Use [REDACTED:API_KEY] for billing."
        )


@pytest.mark.parametrize("stream", [False, True])
async def test_personal_data_in_the_answer_is_redacted(gateway, answer, stream):
    answer.content = f"Anna's PESEL is {PESEL}, mail anna@firma.pl."
    response = await ask(gateway, "Who is the customer?", stream=stream)
    assert response.status_code == 200
    assert PESEL not in response.text
    assert "anna@firma.pl" not in response.text
    if not stream:
        assert response.json()["choices"][0]["message"]["content"] == (
            "Anna's PESEL is [REDACTED:PL_PESEL], mail [REDACTED:EMAIL_ADDRESS]."
        )
    assert_nothing_leaked(gateway, PESEL, "anna@firma.pl")


async def test_pii_block_mode_refuses_the_prompt(gateway, answer):
    edit_policy(gateway, pii={"mode": "block"})
    response = await ask(gateway, f"PESEL {PESEL}")
    assert (response.status_code, response.json()["error"]["code"]) == (403, "pii_detected")
    assert not answer.route.called


async def test_unconfigured_entities_pass(gateway, answer):
    edit_policy(gateway, pii={"mode": "redact", "entities": ["EMAIL_ADDRESS"]})
    await ask(gateway, f"PESEL {PESEL}, mail jan@firma.pl")
    assert sent_content(answer.route) == f"PESEL {PESEL}, mail [REDACTED:EMAIL_ADDRESS]"


async def test_permissive_profile_logs_pii_but_secrets_still_enforce(gateway, answer):
    # Drop the explicit modes so the profile decides; secrets is mandatory and ignores it.
    edit_policy(gateway, profile="permissive", pii={"threshold": 0.6}, secrets={})
    response = await ask(gateway, f"PESEL {PESEL}")
    assert response.status_code == 200
    assert sent_content(answer.route) == f"PESEL {PESEL}"  # recorded, not applied
    (entry,) = gateway.audit_entries()
    assert entry["decision"] == "allow"
    assert verdict(entry, "pii", "pre")["enforced"] is False
    assert entry["risk"] == pytest.approx(0.1)  # the detection still counts

    blocked = await ask(gateway, f"key {API_KEY}")
    assert (blocked.status_code, blocked.json()["error"]["code"]) == (403, "secret_detected")
    assert answer.route.call_count == 1
    assert_nothing_leaked(gateway, PESEL, API_KEY)


# ------------------------------------------------------------------ review evasions


async def test_secret_in_a_tool_definition_never_reaches_the_upstream(gateway, answer):
    tool = {"type": "function", "function": {"name": "pay", "description": f"key {STRIPE}"}}
    response = await ask(gateway, "Pay the invoice.", tools=[tool])
    assert (response.status_code, response.json()["error"]["code"]) == (403, "secret_detected")
    assert not answer.route.called


@pytest.mark.parametrize("stream", [False, True])
async def test_legacy_function_call_with_an_escaped_key_is_masked_in_json_and_sse(
    gateway, llm_upstream, stream
):
    edit_policy(gateway, secrets={"mode": "redact"})
    escaped = json.dumps({"token": STRIPE}).replace("sk_live_", "\\u0073k_live_")
    body = completion(None)
    body["choices"][0]["message"]["function_call"] = {"name": "pay", "arguments": escaped}
    llm_upstream.post("/chat/completions").mock(return_value=httpx.Response(200, json=body))
    response = await ask(gateway, "Pay.", stream=stream)
    assert response.status_code == 200
    assert "function_call" in response.text
    assert "4eC39HqLyjWDarjtT1zdp7dc" not in response.text
    assert "[REDACTED:" in response.text


async def test_unparseable_escaped_tool_arguments_are_refused(gateway, llm_upstream):
    call = {"id": "c", "type": "function", "function": {"name": "f", "arguments": '{"a": "\\u00'}}
    llm_upstream.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=completion(None, tool_calls=[call]))
    )
    response = await ask(gateway, "Call it.")
    assert (response.status_code, response.json()["error"]["code"]) == (
        403,
        "unscannable_content",
    )


async def test_zero_width_split_pesel_is_masked_before_the_upstream(gateway, answer):
    await ask(gateway, "PESEL 4405\u200b1401359, ok?")
    assert sent_content(answer.route) == "PESEL [REDACTED:PL_PESEL], ok?"
