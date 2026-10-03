"""Evasions of ``pii`` and ``secrets`` found in review, each pinned by the control's verdict and
by the document the pipeline would release after applying its redactions.

1. secrets in LLM ``tools`` definitions;          6. quoted passwords redacted only in part;
2. JSON-escaped tool-call arguments;              7. placeholder exemption too broad;
3. values under a credential key (``password``);  8. a value split across parts or messages;
4. MCP embedded resources and base64 blobs;       9. numbers (``{"pesel": 44051401359}``);
5. legacy ``function_call`` arguments (SSE);     10. full-width digits, zero-width characters.
"""

import base64
import json
from typing import Any

import pytest

from gateway.controls.pii import PiiControl
from gateway.controls.secrets import SecretScanner, SecretsControl
from gateway.core.envelope import Interaction, Verdict
from gateway.core.interfaces import Control, ControlConfig
from gateway.core.types import Action, Channel, ControlMode, Decision, Stage
from gateway.policy.schema import PiiConfig
from gateway.proxies.llm import sse_events
from gateway.redaction import apply_redactions

PII, SECRETS = PiiControl(), SecretsControl()
REDACT, BLOCK = ControlMode.REDACT, ControlMode.BLOCK
PESEL = "44051401359"
STRIPE = "sk_live_" + "4eC39HqLyjWDarjtT1zdp7dc"  # fake, assembled
OPENAI = "sk-proj-" + "Ab3De5Fg7Hi9Jk1Lm3No5Pq7Rs9Tu1Vw3Xy5Za7Bc9De"
ZWSP, WORD_JOINER, SOFT_HYPHEN = "\u200b", "\u2060", "\u00ad"  # format characters (Cf)


def full_width(text: str) -> str:
    """ASCII letters, digits and ``_`` as their full-width forms (U+FF01..U+FF5E)."""
    return "".join(chr(ord(c) + 0xFEE0) for c in text)


def cfg(control: Control, mode: ControlMode) -> ControlConfig:
    return PiiConfig(mode=mode) if control is PII else ControlConfig(mode=mode)


async def evaluate(
    make_ctx,
    control: Control,
    channel: Channel,
    stage: Stage,
    document: Any,
    mode: ControlMode = REDACT,
) -> tuple[Verdict, Any]:
    """The verdict, and the stage document after its redactions (as the pipeline applies them)."""
    llm = channel is Channel.LLM
    payload = document if stage is Stage.PRE else {"name": "t", "arguments": {}}
    item = Interaction(
        session_id="s-test",
        principal="anna@demo",
        actor="databot",
        mode=make_ctx().mode,
        channel=channel,
        action=Action.GENERATE if llm else Action.READ,
        resource="model:qwen3:8b" if llm else "web:example.com",
        payload=payload,
        result=None if stage is Stage.PRE else document,
        context=make_ctx(),
    )
    verdict = await control.evaluate(item, stage, cfg(control, mode))
    redacted = apply_redactions(document, verdict.redactions) if verdict.enforced else document
    return verdict, redacted


def chat(*messages: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"model": "qwen3:8b", "messages": list(messages), **extra}


def completion(**message: Any) -> dict[str, Any]:
    return {
        "id": "c",
        "object": "chat.completion",
        "created": 1,
        "model": "qwen3:8b",
        "choices": [{"index": 0, "message": {"role": "assistant", **message}}],
    }


def tool_call(arguments: str) -> dict[str, Any]:
    return {"id": "c1", "type": "function", "function": {"name": "f", "arguments": arguments}}


# --------------------------------------------------------------------- 1. tools


@pytest.mark.parametrize(
    "tool",
    [
        {"type": "function", "function": {"name": "f", "description": f"use key {STRIPE}"}},
        {
            "type": "function",
            "function": {"name": "f", "parameters": {"properties": {"k": {"default": STRIPE}}}},
        },
    ],
    ids=["description", "schema-default"],
)
async def test_1_secret_in_a_tool_definition_is_found(make_ctx, tool):
    request = chat({"role": "user", "content": "hi"}, tools=[tool])
    verdict, _ = await evaluate(make_ctx, SECRETS, Channel.LLM, Stage.PRE, request, BLOCK)
    assert (verdict.decision, verdict.reason_code) == (Decision.BLOCK, "secret_detected")


async def test_1_personal_data_in_a_tool_description_is_redacted_in_place(make_ctx):
    tool = {"type": "function", "function": {"name": "f", "description": f"for PESEL {PESEL}"}}
    request = chat({"role": "user", "content": "hi"}, tools=[tool])
    verdict, redacted = await evaluate(make_ctx, PII, Channel.LLM, Stage.PRE, request)
    assert verdict.decision is Decision.REDACT
    assert redacted["tools"][0]["function"]["description"] == "for PESEL [REDACTED:PL_PESEL]"


# ------------------------------------------------------------ 2. escaped arguments


ESCAPED = json.dumps({"token": STRIPE}).replace("sk_live_", "\\u0073k_live_")


async def test_2_escaped_secret_in_tool_call_arguments_is_found_and_masked_decoded(make_ctx):
    answer = completion(content=None, tool_calls=[tool_call(ESCAPED)])
    verdict, _ = await evaluate(make_ctx, SECRETS, Channel.LLM, Stage.POST, answer, BLOCK)
    assert verdict.decision is Decision.BLOCK
    verdict, redacted = await evaluate(make_ctx, SECRETS, Channel.LLM, Stage.POST, answer)
    arguments = redacted["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(arguments)["token"].startswith("[REDACTED:")  # still valid JSON
    assert STRIPE not in json.loads(arguments)["token"]


@pytest.mark.parametrize(
    ("arguments", "decision", "reason_code"),
    [
        # an escape could hide anything: refused even in redact mode
        ('{"token": "\\u0073k_live_4eC39HqLyjWDarjtT1zdp7dc"', Decision.BLOCK,
         "unscannable_content"),
        # no escape, so the raw text is the content: scanned and masked as written
        ('{"q": "sk_live_4eC39HqLyjWDarjtT1zdp7dc"', Decision.REDACT, "secret_detected"),
    ],
    ids=["broken-with-escape", "broken-plain"],
)  # fmt: skip
async def test_2_unparseable_arguments_fail_closed(make_ctx, arguments, decision, reason_code):
    answer = completion(content=None, tool_calls=[tool_call(arguments)])
    verdict, _ = await evaluate(make_ctx, SECRETS, Channel.LLM, Stage.POST, answer)
    assert (verdict.decision, verdict.reason_code) == (decision, reason_code)


async def test_2_clean_broken_arguments_without_escapes_pass(make_ctx):
    answer = completion(content=None, tool_calls=[tool_call('{"q": "weather in Kraków"')])
    verdict, _ = await evaluate(make_ctx, SECRETS, Channel.LLM, Stage.POST, answer)
    assert verdict.decision is Decision.ALLOW


# ---------------------------------------------------------------- 3. keyed values


@pytest.mark.parametrize(
    ("channel", "stage", "document", "path", "label"),
    [
        (Channel.MCP, Stage.PRE, {"password": "Abc12345!"}, ("password",), "PASSWORD"),
        (Channel.MCP, Stage.PRE, {"db": {"X-Api-Key": "zz99yy88"}}, ("db", "X-Api-Key"),
         "API_KEY"),
        (Channel.MCP, Stage.PRE, {"authorization": "Bearer abcdef"}, ("authorization",),
         "ACCESS_TOKEN"),
        (Channel.MCP, Stage.POST, {"content": [], "structuredContent": {"client_secret": "qw12"
         "erty"}}, ("structuredContent", "client_secret"), "SECRET"),
    ],
    ids=["password", "nested-api-key", "authorization", "structured-content"],
)  # fmt: skip
async def test_3_value_under_a_credential_key_is_masked(
    make_ctx, channel, stage, document, path, label
):
    if stage is Stage.PRE:
        document = {"name": "t", "arguments": document}
        path = ("arguments", *path)
    verdict, redacted = await evaluate(make_ctx, SECRETS, channel, stage, document)
    assert verdict.decision is Decision.REDACT
    for key in path:
        redacted = redacted[key]
    assert redacted == f"[REDACTED:{label}]"


async def test_3_keyed_value_in_decoded_llm_arguments(make_ctx):
    request = chat({"role": "assistant", "tool_calls": [tool_call('{"pwd": "hunter2x"}')]})
    _, redacted = await evaluate(make_ctx, SECRETS, Channel.LLM, Stage.PRE, request)
    arguments = redacted["messages"][0]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(arguments) == {"pwd": "[REDACTED:PASSWORD]"}


@pytest.mark.parametrize(
    "arguments",
    [
        {"password": "${DB_PASSWORD}"},
        {"password": "short"},
        {"token_count": 12345678},
        {"tokens": "a b c d e f"},
        {"passwordHint": "your first pet"},
    ],
)
async def test_3_placeholders_short_values_and_lookalike_keys_pass(make_ctx, arguments):
    document = {"name": "t", "arguments": arguments}
    verdict, _ = await evaluate(make_ctx, SECRETS, Channel.MCP, Stage.PRE, document)
    assert verdict.decision is Decision.ALLOW


# ------------------------------------------------------ 4. resources and blobs


def b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


async def test_4_embedded_resource_text_is_redacted_even_in_an_error(make_ctx):
    result = {
        "content": [{"type": "resource", "resource": {"uri": "r", "text": f"PESEL {PESEL}"}}],
        "isError": True,
    }
    verdict, redacted = await evaluate(make_ctx, PII, Channel.MCP, Stage.POST, result)
    assert verdict.decision is Decision.REDACT
    assert redacted["content"][0]["resource"]["text"] == "PESEL [REDACTED:PL_PESEL]"


@pytest.mark.parametrize("mime", ["text/plain", "image/png"])  # the claimed type does not matter
async def test_4_text_blob_is_decoded_and_blocks_because_no_mask_fits(make_ctx, mime):
    blob = {"uri": "r", "blob": b64(f"key={STRIPE}"), "mimeType": mime}
    result = {"content": [{"type": "resource", "resource": blob}]}
    verdict, _ = await evaluate(make_ctx, SECRETS, Channel.MCP, Stage.POST, result)  # redact mode
    assert (verdict.decision, verdict.reason_code) == (Decision.BLOCK, "secret_detected")


async def test_4_a_key_that_is_valid_base64_is_not_hidden_by_a_blob_field(make_ctx):
    """``AKIA...`` decodes to binary; short payloads are scanned as written too."""
    key = "AKIA" + "Z7QW3ERT5YUI2OPA"
    document = {"name": "t", "arguments": {"blob": key}}
    verdict, redacted = await evaluate(make_ctx, SECRETS, Channel.MCP, Stage.PRE, document)
    assert verdict.decision is Decision.REDACT
    assert redacted["arguments"]["blob"] == "[REDACTED:API_KEY]"


async def test_4_binary_blob_is_left_alone(make_ctx):
    data = base64.b64encode(b"\x89PNG\r\n\x1a\n\x00\x00" + STRIPE.encode()).decode()
    result = {"content": [{"type": "image", "data": data, "mimeType": "image/png"}]}
    verdict, _ = await evaluate(make_ctx, SECRETS, Channel.MCP, Stage.POST, result)
    assert verdict.decision is Decision.ALLOW


# ---------------------------------------------------------- 5. legacy function_call


async def test_5_legacy_function_call_is_scanned_and_sse_carries_only_the_mask(make_ctx):
    answer = completion(content=None, function_call={"name": "f", "arguments": ESCAPED})
    verdict, _ = await evaluate(make_ctx, SECRETS, Channel.LLM, Stage.POST, answer, BLOCK)
    assert verdict.decision is Decision.BLOCK
    verdict, redacted = await evaluate(make_ctx, SECRETS, Channel.LLM, Stage.POST, answer)
    streamed = b"".join(sse_events(redacted)).decode()
    assert "function_call" in streamed
    assert "4eC39HqLyjWDarjtT1zdp7dc" not in streamed
    assert "[REDACTED:ACCESS_TOKEN+API_KEY]" in streamed  # keyed "token" and a Stripe key


# ------------------------------------------------------------- 6. quoted values


@pytest.mark.parametrize(
    ("text", "secret"),
    [
        ('password="Abc12345 secret-tail" next', "Abc12345 secret-tail"),
        ("password='Abc12345 secret-tail'", "Abc12345 secret-tail"),
        ('password="Abc\\"123 tail" next', 'Abc\\"123 tail'),  # escaped quote stays inside
        ('{"api_key": "abcd 1234 efgh"}', "abcd 1234 efgh"),
        ('password="Unterminated 2nd half', "Unterminated 2nd half"),
    ],
)
def test_6_a_quoted_value_is_taken_whole(text, secret):
    (finding,) = SecretScanner().find(text)
    assert text[finding.start : finding.end] == secret


# ---------------------------------------------------------------- 7. placeholders


@pytest.mark.parametrize(
    ("text", "found"),
    [
        ("password=%Xy7#pq9z", True),  # starts with % but is no %VAR% expression
        ("password=MyGetenvPass1", True),  # contains getenv but is no reference
        ('password="${A}tail9X"', True),  # a template plus more is a value
        ("postgres://u:p%40ssW0rd1@db/x", True),  # decodes to p@ssW0rd1
        ("postgres://u:%24%7BDB_PASSWORD%7D@db/x", False),  # decodes to ${DB_PASSWORD}
        ('api_key="${API_KEY}"', False),
        ("password=%DB_PASSWORD%", False),
        ("password={{ db_password }}", False),
        ("password=<your-password>", False),
        ("api_key=os.environ", False),
    ],
)
def test_7_only_complete_placeholders_are_exempt(text, found):
    assert bool(SecretScanner().find(text)) is found


# --------------------------------------------------------------- 8. split values


async def test_8_secret_split_across_content_parts_is_masked_in_every_part(make_ctx):
    parts = [{"type": "text", "text": f"key {OPENAI[:14]}"}, {"type": "text", "text": OPENAI[14:]}]
    request = chat({"role": "user", "content": parts})
    verdict, _ = await evaluate(make_ctx, SECRETS, Channel.LLM, Stage.PRE, request, BLOCK)
    assert verdict.decision is Decision.BLOCK
    verdict, redacted = await evaluate(make_ctx, SECRETS, Channel.LLM, Stage.PRE, request)
    texts = [part["text"] for part in redacted["messages"][0]["content"]]
    assert texts == ["key [REDACTED:API_KEY]", "[REDACTED:API_KEY]"]


async def test_8_pesel_split_across_messages_is_masked_in_both(make_ctx):
    request = chat(
        {"role": "user", "content": "mój PESEL to 44051"}, {"role": "user", "content": "401359"}
    )
    verdict, redacted = await evaluate(make_ctx, PII, Channel.LLM, Stage.PRE, request)
    assert verdict.decision is Decision.REDACT
    assert [m["content"] for m in redacted["messages"]] == [
        "mój PESEL to [REDACTED:PL_PESEL]",
        "[REDACTED:PL_PESEL]",
    ]


async def test_8_split_into_a_blob_cannot_be_masked_and_blocks(make_ctx):
    result = {
        "content": [
            {"type": "text", "text": OPENAI[:14]},
            {"type": "resource", "resource": {"uri": "r", "blob": b64(OPENAI[14:])}},
        ]
    }
    verdict, _ = await evaluate(make_ctx, SECRETS, Channel.MCP, Stage.POST, result)
    assert verdict.decision is Decision.BLOCK


# ------------------------------------------------------------------- 9. numbers


@pytest.mark.parametrize("value", [44051401359, 44051401359.0])
async def test_9_a_number_is_inspected_and_replaced_whole(make_ctx, value):
    document = {"name": "t", "arguments": {"pesel": value}}
    verdict, redacted = await evaluate(make_ctx, PII, Channel.MCP, Stage.PRE, document)
    assert verdict.decision is Decision.REDACT
    assert redacted["arguments"]["pesel"] == "[REDACTED:PL_PESEL]"
    verdict, _ = await evaluate(make_ctx, PII, Channel.MCP, Stage.PRE, document, BLOCK)
    assert verdict.decision is Decision.BLOCK


async def test_9_numbers_in_structured_results_too(make_ctx):
    result = {"content": [], "structuredContent": {"rows": [{"nip": 1234563218}]}}
    _, redacted = await evaluate(make_ctx, PII, Channel.MCP, Stage.POST, result)
    assert redacted["structuredContent"]["rows"][0]["nip"] == "[REDACTED:PL_NIP]"


# --------------------------------------------------------- 10. disguised characters


@pytest.mark.parametrize(
    ("control", "text", "masked"),
    [
        (PII, f"PESEL {full_width(PESEL)} koniec", "PESEL [REDACTED:PL_PESEL] koniec"),
        (PII, f"PESEL 4405{ZWSP}1401359 koniec", "PESEL [REDACTED:PL_PESEL] koniec"),
        (PII, f"PESEL 44051{WORD_JOINER}401359{SOFT_HYPHEN}.",
         f"PESEL [REDACTED:PL_PESEL]{SOFT_HYPHEN}."),
        (SECRETS, f"key {STRIPE[:10]}{ZWSP}{STRIPE[10:]}!", "key [REDACTED:API_KEY]!"),
        (SECRETS, f"key {full_width('sk_live_')}{STRIPE[8:]}", "key [REDACTED:API_KEY]"),
    ],
    ids=["full-width", "zero-width-space", "word-joiner", "zero-width-in-key", "full-width-key"],
)  # fmt: skip
async def test_10_disguised_characters_are_seen_and_the_whole_range_masked(
    make_ctx, control, text, masked
):
    request = chat({"role": "user", "content": text})
    verdict, redacted = await evaluate(make_ctx, control, Channel.LLM, Stage.PRE, request)
    assert verdict.decision is Decision.REDACT
    assert redacted["messages"][0]["content"] == masked
