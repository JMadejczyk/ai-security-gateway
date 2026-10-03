"""TextExtractor: every scannable value, with a pointer `apply_redactions` can apply."""

import base64
import json
from typing import Any

import pytest

from gateway.controls import text as text_module
from gateway.controls.text import (
    SegmentKind,
    TextExtractor,
    TextSegment,
    decode_blob,
    pointer,
    string_leaves,
)
from gateway.core.envelope import Interaction
from gateway.core.types import Action, Channel, Stage
from gateway.redaction import apply_redactions

EXTRACTOR = TextExtractor()
TEXT, NUMBER, OPAQUE = SegmentKind.TEXT, SegmentKind.NUMBER, SegmentKind.OPAQUE
UNSCANNABLE = SegmentKind.UNSCANNABLE


def b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


PNG = base64.b64encode(b"\x89PNG\x00" * 800).decode()  # binary media, longer than any key


CHAT_REQUEST: dict[str, Any] = {
    "model": "m",
    "temperature": 0.2,
    "messages": [
        {"role": "system", "content": "You are DataBot."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}},
                {"type": "image_url", "image_url": {"url": f"data:text/plain;base64,{b64('hi')}"}},
            ],
        },
        {
            "role": "assistant",
            "reasoning": "query it",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "query",
                        "arguments": '{"sql": "SELECT 1", "n": 44051401359, "k": "\\u0041B"}',
                    },
                }
            ],
            "function_call": {"name": "legacy", "arguments": '{"q": "old"}'},
        },
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "query",
                "description": "Run SQL",
                "parameters": {"properties": {"sql": {"default": "SELECT 2"}}},
            },
        }
    ],
}

COMPLETION: dict[str, Any] = {
    "id": "x",
    "created": 1_790_000_000,
    "choices": [
        {
            "message": {
                "content": "Hi",
                "tool_calls": [{"function": {"arguments": "not json"}}],
                "reasoning_details": [{"text": "think"}],
            },
            "logprobs": {"content": [{"token": "H"}]},
        }
    ],
}

TOOL_CALL: dict[str, Any] = {
    "name": "fetch",
    "arguments": {
        "url": "https://example.com",
        "a/b": "slash key",
        "m~n": 3,
        "flag": True,
        "headers": [{"k": "accept"}, "plain"],
    },
}

TOOL_RESULT: dict[str, Any] = {
    "content": [
        {"type": "text", "text": "page"},
        {"type": "image", "data": PNG, "mimeType": "image/png"},
        {"type": "resource", "resource": {"uri": "file:///r", "text": "inner"}},
        {"type": "resource", "resource": {"uri": "u", "blob": b64("blob text"), "mimeType": "x"}},
    ],
    "structuredContent": {"pesel": 44051401359, "ratio": 0.5},
    "isError": True,
}


def interaction(make_ctx, channel: Channel, *, payload: Any = None, result: Any = None):
    llm = channel is Channel.LLM
    return Interaction(
        session_id="s-test",
        principal="anna@demo",
        actor="databot",
        mode=make_ctx().mode,
        channel=channel,
        action=Action.GENERATE if llm else Action.READ,
        resource="model:qwen3:8b" if llm else "web:example.com",
        payload=payload,
        result=result,
        context=make_ctx(),
    )


def extract(make_ctx, channel, stage, payload, result=None) -> list[TextSegment]:
    return EXTRACTOR.segments(interaction(make_ctx, channel, payload=payload, result=result), stage)


def test_llm_request_every_string_tools_included_arguments_decoded(make_ctx):
    found = extract(make_ctx, Channel.LLM, Stage.PRE, CHAT_REQUEST)
    args = "/messages/2/tool_calls/0/function/arguments"
    assert [(s.pointer, s.embedded, s.text, s.kind) for s in found] == [
        ("/model", None, "m", TEXT),
        ("/messages/0/role", None, "system", TEXT),
        ("/messages/0/content", None, "You are DataBot.", TEXT),
        ("/messages/1/role", None, "user", TEXT),
        ("/messages/1/content/0/type", None, "text", TEXT),
        ("/messages/1/content/0/text", None, "Look:", TEXT),
        ("/messages/1/content/1/type", None, "image_url", TEXT),
        # the PNG data URL is binary: not scanned
        ("/messages/1/content/2/type", None, "image_url", TEXT),
        ("/messages/1/content/2/image_url/url", None, "hi", OPAQUE),
        ("/messages/2/role", None, "assistant", TEXT),
        ("/messages/2/reasoning", None, "query it", TEXT),
        ("/messages/2/tool_calls/0/id", None, "c1", TEXT),
        ("/messages/2/tool_calls/0/function/name", None, "query", TEXT),
        (args, "/sql", "SELECT 1", TEXT),
        (args, "/n", "44051401359", NUMBER),
        (args, "/k", "AB", TEXT),  # A decoded
        ("/messages/2/function_call/name", None, "legacy", TEXT),
        ("/messages/2/function_call/arguments", "/q", "old", TEXT),
        ("/tools/0/type", None, "function", TEXT),
        ("/tools/0/function/name", None, "query", TEXT),
        ("/tools/0/function/description", None, "Run SQL", TEXT),
        ("/tools/0/function/parameters/properties/sql/default", None, "SELECT 2", TEXT),
    ]


def test_llm_completion_every_string_numbers_skipped(make_ctx):
    found = extract(make_ctx, Channel.LLM, Stage.POST, CHAT_REQUEST, COMPLETION)
    assert [(s.pointer, s.embedded, s.text, s.kind) for s in found] == [
        ("/id", None, "x", TEXT),
        ("/choices/0/message/content", None, "Hi", TEXT),
        ("/choices/0/message/tool_calls/0/function/arguments", None, "not json", TEXT),
        ("/choices/0/message/reasoning_details/0/text", None, "think", TEXT),
        ("/choices/0/logprobs/content/0/token", None, "H", TEXT),
    ]


def test_mcp_arguments_strings_and_numbers_with_their_keys(make_ctx):
    found = extract(make_ctx, Channel.MCP, Stage.PRE, TOOL_CALL)
    assert [(s.pointer, s.key, s.text, s.kind) for s in found] == [
        ("/arguments/url", "url", "https://example.com", TEXT),
        ("/arguments/a~1b", "a/b", "slash key", TEXT),
        ("/arguments/m~0n", "m~n", "3", NUMBER),
        ("/arguments/headers/0/k", "k", "accept", TEXT),
        ("/arguments/headers/1", "headers", "plain", TEXT),  # list items inherit the key
    ]


def test_mcp_result_resources_blobs_numbers_even_on_errors(make_ctx):
    found = extract(make_ctx, Channel.MCP, Stage.POST, TOOL_CALL, TOOL_RESULT)
    assert [(s.pointer, s.text, s.kind) for s in found] == [
        ("/content/0/type", "text", TEXT),
        ("/content/0/text", "page", TEXT),
        ("/content/1/type", "image", TEXT),
        # the PNG payload is binary: not scanned
        ("/content/1/mimeType", "image/png", TEXT),
        ("/content/2/type", "resource", TEXT),
        ("/content/2/resource/uri", "file:///r", TEXT),
        ("/content/2/resource/text", "inner", TEXT),
        ("/content/3/type", "resource", TEXT),
        ("/content/3/resource/uri", "u", TEXT),
        ("/content/3/resource/blob", "blob text", OPAQUE),  # text, whatever the mimeType says
        ("/content/3/resource/mimeType", "x", TEXT),
        ("/structuredContent/pesel", "44051401359", NUMBER),
        ("/structuredContent/ratio", "0.5", NUMBER),
    ]


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ('{"a": "x"}', [("/a", "x", TEXT)]),
        ('"bare"', [("", "bare", TEXT)]),
        ("[1.0, 2e3]", [("/0", "1", NUMBER), ("/1", "2000", NUMBER)]),
        ('{"a": "x"', [(None, '{"a": "x"', TEXT)]),  # broken, but nothing can hide
        ('{"a": "\\u0041KIA', [(None, '{"a": "\\u0041KIA', UNSCANNABLE)]),  # broken + escape
    ],
)
def test_tool_call_arguments(make_ctx, arguments, expected):
    call = {"function": {"arguments": arguments}}
    request = {"messages": [{"role": "assistant", "tool_calls": [call]}]}
    found = [s for s in extract(make_ctx, Channel.LLM, Stage.PRE, request) if s.key != "role"]
    assert [(s.embedded, s.text, s.kind) for s in found] == expected


@pytest.mark.parametrize(
    ("encoded", "is_base64", "expected"),
    [
        (b64("Zażółć"), True, (OPAQUE, "Zażółć")),
        (PNG, True, None),  # binary media
        ("AKIAZ7QW3ERT5YUI2OPA", True, (TEXT, "AKIAZ7QW3ERT5YUI2OPA")),  # short: scanned raw
        ("not base64!", True, (TEXT, "not base64!")),
        ("hello%20world", False, (OPAQUE, "hello world")),
    ],
)
def test_decode_blob(encoded, is_base64, expected):
    assert decode_blob(encoded, is_base64=is_base64) == expected


def test_oversized_text_blob_is_unscannable_and_binary_is_left_alone(monkeypatch):
    monkeypatch.setattr(text_module, "MAX_DECODED_BYTES", 16)
    assert decode_blob(b64("x" * 64)) == (UNSCANNABLE, b64("x" * 64))
    assert decode_blob(base64.b64encode(b"\x00" * 64).decode()) is None


@pytest.mark.parametrize(
    ("channel", "stage", "payload", "result"),
    [
        (Channel.LLM, Stage.PRE, CHAT_REQUEST, None),
        (Channel.LLM, Stage.POST, CHAT_REQUEST, COMPLETION),
        (Channel.MCP, Stage.PRE, TOOL_CALL, None),
        (Channel.MCP, Stage.POST, TOOL_CALL, TOOL_RESULT),
    ],
)
def test_every_redactable_segment_masks_exactly_its_value(
    make_ctx, channel, stage, payload, result
):
    """A span over each whole redactable segment lands on that value and nowhere else."""
    item = interaction(make_ctx, channel, payload=payload, result=result)
    document = EXTRACTOR.document(item, stage)
    segments = [s for s in EXTRACTOR.segments(item, stage) if s.redactable]
    redacted = apply_redactions(document, [s.span(0, len(s.text), "X") for s in segments])
    field = "payload" if stage is Stage.PRE else "result"
    again = EXTRACTOR.segments(item.model_copy(update={field: redacted}), stage)
    assert [s.text for s in again if s.redactable] == ["[REDACTED:X]"] * len(segments)
    assert document == (payload if stage is Stage.PRE else result)  # the input is untouched


def test_redacted_arguments_stay_valid_json(make_ctx):
    item = interaction(make_ctx, Channel.LLM, payload=CHAT_REQUEST)
    (sql,) = [s for s in EXTRACTOR.segments(item, Stage.PRE) if s.embedded == "/sql"]
    redacted: Any = apply_redactions(CHAT_REQUEST, [sql.span(7, 8, "N")])
    arguments = redacted["messages"][2]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(arguments) == {"sql": "SELECT [REDACTED:N]", "n": 44051401359, "k": "AB"}


@pytest.mark.parametrize(
    ("payload", "stage", "result"),
    [(["not", "a", "request"], Stage.PRE, None), (CHAT_REQUEST, Stage.POST, None)],
)
def test_odd_shapes_yield_nothing(make_ctx, payload, stage, result):
    assert extract(make_ctx, Channel.LLM, stage, payload, result) == []
    assert extract(make_ctx, Channel.A2A, Stage.PRE, CHAT_REQUEST) == []


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        ((), ""),
        (("messages", 0, "content"), "/messages/0/content"),
        (("a/b", "m~n"), "/a~1b/m~0n"),
        (("~1",), "/~01"),  # escape ~ first, so a literal "~1" survives the round trip
    ],
)
def test_pointer_escaping(tokens, expected):
    assert pointer(*tokens) == expected


def test_string_leaves_stays_strings_only():
    """`signatures` reads it for path globs: strings only, unchanged."""
    assert list(string_leaves("text")) == [TextSegment("", "text")]
    assert list(string_leaves({"a": 1, "b": ["x"]})) == [TextSegment("/b/0", "x")]
    assert list(string_leaves("")) == []
