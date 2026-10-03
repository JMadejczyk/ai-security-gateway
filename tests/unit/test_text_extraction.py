"""TextExtractor: every scannable string, with a JSON pointer `apply_redactions` can apply."""

from typing import Any

import pytest

from gateway.controls.text import TextExtractor, TextSegment, pointer, string_leaves
from gateway.core.envelope import Interaction, Span
from gateway.core.types import Action, Channel, Stage
from gateway.redaction import apply_redactions

EXTRACTOR = TextExtractor()

CHAT_REQUEST: dict[str, Any] = {
    "model": "qwen3:8b",
    "messages": [
        {"role": "system", "content": "You are DataBot."},
        {
            "role": "user",
            "name": "anna",
            "content": [
                {"type": "text", "text": "Look at this:"},
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                {"type": "text", "text": "and this."},
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "reasoning": "I should query.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "query", "arguments": '{"sql": "SELECT 1"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "1 row"},
        {"role": "user", "content": ""},
    ],
}

COMPLETION: dict[str, Any] = {
    "id": "chatcmpl-1",
    "model": "qwen3:8b",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "There are 40."}},
        {
            "index": 1,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c", "function": {"name": "fetch", "arguments": '{"url": "x"}'}},
                    {"id": "d", "function": {"name": "noop", "arguments": ""}},
                ],
            },
        },
    ],
    "usage": {"total_tokens": 3},
}

TOOL_CALL: dict[str, Any] = {
    "name": "fetch",
    "arguments": {
        "url": "https://example.com",
        "a/b": "slash key",
        "m~n": "tilde key",
        "limit": 5,
        "headers": [{"k": "accept", "v": ""}, "plain"],
    },
}

TOOL_RESULT: dict[str, Any] = {
    "content": [
        {"type": "text", "text": "page body"},
        {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"},
        {"type": "text", "text": "second"},
    ],
    "structuredContent": {"rows": [{"name": "Anna", "n": 3}], "note": "ok"},
    "isError": False,
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


@pytest.mark.parametrize(
    ("channel", "stage", "payload", "result", "expected"),
    [
        pytest.param(
            Channel.LLM,
            Stage.PRE,
            CHAT_REQUEST,
            None,
            [
                ("/messages/0/content", "You are DataBot."),
                ("/messages/1/content/0/text", "Look at this:"),
                ("/messages/1/content/2/text", "and this."),
                ("/messages/2/reasoning", "I should query."),
                ("/messages/2/tool_calls/0/function/arguments", '{"sql": "SELECT 1"}'),
                ("/messages/3/content", "1 row"),
            ],
            id="llm-pre: strings, content parts, prose extras, tool-call arguments",
        ),
        pytest.param(
            Channel.LLM,
            Stage.POST,
            CHAT_REQUEST,
            COMPLETION,
            [
                ("/choices/0/message/content", "There are 40."),
                ("/choices/1/message/tool_calls/0/function/arguments", '{"url": "x"}'),
            ],
            id="llm-post: answer text and tool calls, never the request",
        ),
        pytest.param(
            Channel.MCP,
            Stage.PRE,
            TOOL_CALL,
            None,
            [
                ("/arguments/url", "https://example.com"),
                ("/arguments/a~1b", "slash key"),
                ("/arguments/m~0n", "tilde key"),
                ("/arguments/headers/0/k", "accept"),
                ("/arguments/headers/1", "plain"),
            ],
            id="mcp-pre: string leaves of the arguments, RFC 6901 escaping",
        ),
        pytest.param(
            Channel.MCP,
            Stage.POST,
            TOOL_CALL,
            TOOL_RESULT,
            [
                ("/content/0/text", "page body"),
                ("/content/2/text", "second"),
                ("/structuredContent/rows/0/name", "Anna"),
                ("/structuredContent/note", "ok"),
            ],
            id="mcp-post: text content items and structured content",
        ),
        pytest.param(Channel.LLM, Stage.PRE, ["not", "a", "request"], None, [], id="odd-shape"),
        pytest.param(Channel.LLM, Stage.POST, CHAT_REQUEST, None, [], id="no-result"),
        pytest.param(Channel.MCP, Stage.POST, TOOL_CALL, {"content": "x"}, [], id="odd-result"),
        pytest.param(Channel.A2A, Stage.PRE, CHAT_REQUEST, None, [], id="unknown-channel"),
    ],
)
def test_segments_per_channel_and_stage(make_ctx, channel, stage, payload, result, expected):
    found = EXTRACTOR.segments(
        interaction(make_ctx, channel, payload=payload, result=result), stage
    )
    assert [(s.pointer, s.text) for s in found] == expected


@pytest.mark.parametrize(
    ("channel", "stage", "payload", "result"),
    [
        (Channel.LLM, Stage.PRE, CHAT_REQUEST, None),
        (Channel.LLM, Stage.POST, CHAT_REQUEST, COMPLETION),
        (Channel.MCP, Stage.PRE, TOOL_CALL, None),
        (Channel.MCP, Stage.POST, TOOL_CALL, TOOL_RESULT),
    ],
)
def test_every_pointer_is_redactable_in_the_stage_document(
    make_ctx, channel, stage, payload, result
):
    """A span over each whole segment lands exactly on that string and nowhere else."""
    item = interaction(make_ctx, channel, payload=payload, result=result)
    document = EXTRACTOR.document(item, stage)
    segments = EXTRACTOR.segments(item, stage)
    spans = [Span(path=s.pointer, start=0, end=len(s.text), label="X") for s in segments]
    redacted = apply_redactions(document, spans)
    assert [s.text for s in EXTRACTOR.segments(_with(item, stage, redacted), stage)] == [
        "[REDACTED:X]"
    ] * len(segments)
    assert document == (payload if stage is Stage.PRE else result)  # the input is untouched


def _with(item: Interaction, stage: Stage, document: object) -> Interaction:
    field = "payload" if stage is Stage.PRE else "result"
    return item.model_copy(update={field: document})


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


def test_string_leaves_of_a_bare_string_is_the_document_itself():
    assert list(string_leaves("text")) == [TextSegment("", "text")]
    assert list(string_leaves("")) == []
    assert list(string_leaves(42)) == []
