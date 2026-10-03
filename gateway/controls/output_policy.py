"""``output_policy``: does the model's answer reveal data outside the session's scope?

Post, LLM channel. The judge gets the session's effective scope (permission strings from the
evaluator, plus the agent's own allow and deny lists) as trusted instructions, and the
answer's text as untrusted data, and lists the passages that reveal out-of-scope data as
verbatim quotes. Each quote is mapped back to **exact** spans:

- the answer is every value of the completion except its protocol fields, which are
  excluded by *location* (``/id``, ``/model``, ``/choices/i/message/role``, a tool call's
  ``id`` and ``type``, ...): message content, reasoning, and every string and number inside
  decoded tool-call arguments, whatever its key (an argument called ``name`` or ``id`` is
  data). Arguments values are labelled with their pointer so the judge knows what they are;
- the judge sees the text delimiter-escaped; quotes are matched against exactly that text
  and mapped back to original offsets through an offset map (`JudgedText`), never by
  unescaping; a quote across two values yields a span in each. Every occurrence is a span
  (the same datum repeated is the same leak). A quote found nowhere is ignored: judges
  hallucinate and paraphrase, and a span must point at real text;
- ``redact`` masks the spans (a number is masked whole); ``block`` refuses the answer. A hit
  in decoded base64 text blocks in both modes (no mask fits), and arguments that could not
  be decoded reliably block unjudged (``answer_unscannable``).

**Unavailable fails closed to block in both modes.** ``redact`` needs to know *which* spans
to mask; without a judge the gateway does not know, and masking the whole answer would be a
block with a misleading audit entry. No ``judges`` section: the control is off.

**Limits** (inherently fuzzy, kept conservative): the judge can only judge the scope it is
told and the text it sees, and it cannot know where a fact came from (a public fact that
happens to match a table is not a leak). A leak the judge reports with a paraphrase instead
of a verbatim quote is missed in both modes, because an unmatched quote is ignored. Answers
longer than ``judges.max_content_chars`` are not judged (unavailable → block): raise the
limit or keep answers short. Row-level security and ``authz`` remain the real boundary;
this is a backstop for what the model says.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Final, Self, override

from pydantic import BaseModel, ConfigDict, Field

from gateway.clock import Clock, utc_now
from gateway.controls.scope import current_scope
from gateway.controls.text import SegmentKind, TextExtractor, TextSegment
from gateway.core.envelope import Interaction, Span, Verdict
from gateway.core.interfaces import Control, ControlConfig
from gateway.core.types import ControlKind, ControlMode, Decision, Stage
from gateway.judges.client import (
    JudgeClient,
    JudgeUnavailableError,
    escape_delimiters,
    escape_with_map,
)
from gateway.policy.evaluator import PolicyEvaluator

LABEL: Final = "OUT_OF_SCOPE"
IN_SCOPE: Final = "answer_in_scope"
OUT_OF_SCOPE: Final = "out_of_scope_data"
QUOTES_UNMATCHED: Final = "judge_quotes_unmatched"
UNAVAILABLE: Final = "judge_unavailable"
NO_CONTENT: Final = "no_content"
UNSCANNABLE: Final = "answer_unscannable"
NOT_CONFIGURED: Final = "judge_not_configured"
SEPARATOR: Final = "\n\n"
MAX_VIOLATIONS: Final = 100

INSTRUCTIONS: Final = """\
An AI assistant answered a user in a session that may only access data covered by these \
permissions (format action:namespace:identifier, `*` matches anything):
- session scope: {scope}
- the agent may also only use: {agent_allow}
- never: {agent_deny}
Data is in scope only if the session scope AND the agent's list cover the resource it came \
from (for example `read:db:sales.orders` covers rows of table sales.orders only).

The data block is the assistant's answer. Find passages that reveal data the session could \
not have read under these permissions: records, values or contents of a database table, \
file, web host or other resource that is not covered. General knowledge, the assistant's own \
reasoning and wording, and data from covered resources are NOT violations. When unsure, \
report nothing.

For each violation, copy the revealing passage exactly as it appears, character for \
character, as `quote` (keep it as short as possible while covering the data), and give a \
short `reason` naming the resource without repeating the data. If there is none, return \
{{"violations": []}}."""


class Violation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    quote: str = Field(max_length=4000)
    reason: str = Field(default="", max_length=500)


class OutputAssessment(BaseModel):
    """Strict: ``violations`` is required (``[]`` is the in-scope verdict) and unknown keys
    are refused, so ``{}`` or ``{"error": ...}`` is an unavailable judge, never "allow"."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    violations: list[Violation] = Field(max_length=MAX_VIOLATIONS)


# Protocol fields of a chat completion, by location (never by key name: a tool argument
# called ``name``, ``id`` or ``uri`` is data the model wrote and is judged like any other).
_PROTOCOL_FIELD: Final = re.compile(
    r"/(?:id|object|model|system_fingerprint|service_tier)"
    r"|/choices/\d+/(?:finish_reason|message/role|message/tool_calls/\d+/(?:id|type))"
)
_JUDGED_KINDS: Final = frozenset({SegmentKind.TEXT, SegmentKind.NUMBER, SegmentKind.OPAQUE})


def _is_protocol_field(segment: TextSegment) -> bool:
    return segment.embedded is None and _PROTOCOL_FIELD.fullmatch(segment.pointer) is not None


def _answer_segments(interaction: Interaction) -> list[TextSegment]:
    """Every value of the completion except its protocol fields: message text, reasoning,
    and every string *and number* inside decoded tool-call arguments, decoded base64 text
    included (`SegmentKind.OPAQUE`: judged, but a hit there blocks, a mask cannot fit)."""
    return [
        segment
        for segment in TextExtractor().segments(interaction, Stage.POST)
        if not _is_protocol_field(segment) and segment.text.strip()
    ]


@dataclass(frozen=True, slots=True)
class JudgedText:
    """The content the judge sees, and where each of its characters came from.

    Segments are joined with a blank line; a value inside tool-call arguments is preceded by
    its pointer (``/card = 4111...``) so the judge knows what it is. Each value is delimiter-
    escaped exactly as the judge client would escape it (escaping is idempotent, so the
    client leaves this text as it is). ``origin[k]`` is ``(segment index, original offset)``
    of character ``k``, or None for the separators and labels the gateway added."""

    segments: tuple[TextSegment, ...]
    text: str
    origin: tuple[tuple[int, int] | None, ...]

    @classmethod
    def of(cls, segments: Sequence[TextSegment]) -> Self:
        parts: list[str] = []
        origin: list[tuple[int, int] | None] = []
        for index, segment in enumerate(segments):
            added = (SEPARATOR if index else "") + (
                f"{escape_delimiters(segment.embedded)} = " if segment.embedded else ""
            )
            parts.append(added)
            origin.extend([None] * len(added))
            escaped, offsets = escape_with_map(segment.text)
            parts.append(escaped)
            origin.extend((index, offset) for offset in offsets)
        return cls(tuple(segments), "".join(parts), tuple(origin))

    def locate(self, quote: str) -> list[tuple[int, int, int]]:
        """``(segment index, start, end)`` in original code points for every verbatim
        occurrence of ``quote`` in the judged text, one piece per segment it covers, plus
        every occurrence in a segment's original text (a judge that undid the escaping)."""
        pieces: list[tuple[int, int, int]] = []
        for low, high in _occurrences(self.text, quote):
            covered: dict[int, list[int]] = {}
            for place in self.origin[low:high]:
                if place is not None:
                    covered.setdefault(place[0], []).append(place[1])
            pieces.extend((i, min(offs), max(offs) + 1) for i, offs in covered.items())
        for index, segment in enumerate(self.segments):
            pieces.extend((index, a, b) for a, b in _occurrences(segment.text, quote))
        return pieces


def _occurrences(text: str, quote: str) -> list[tuple[int, int]]:
    """Every non-overlapping verbatim occurrence of ``quote`` in ``text``."""
    found: list[tuple[int, int]] = []
    start = text.find(quote)
    while start != -1:
        found.append((start, start + len(quote)))
        start = text.find(quote, start + len(quote))
    return found


def quote_spans(judged: JudgedText, quotes: Sequence[str]) -> tuple[tuple[Span, ...], bool]:
    """Spans over every occurrence of each quote (whitespace around it ignored, an empty one
    matches nothing), and whether any lands in text no mask can be written into (decoded
    base64)."""
    spans: list[Span] = []
    opaque = False
    for raw in quotes:
        quote = raw.strip()
        if not quote:
            continue
        for index, start, end in judged.locate(quote):
            segment = judged.segments[index]
            opaque |= not segment.redactable
            spans.append(segment.span(start, end, LABEL))
    return tuple(dict.fromkeys(spans)), opaque


class OutputPolicyControl(Control):
    id: ClassVar[str] = "output_policy"
    stages: ClassVar[frozenset[Stage]] = frozenset({Stage.POST})
    kind: ClassVar[ControlKind] = ControlKind.SEMANTIC

    def __init__(
        self, judge: JudgeClient, evaluator: PolicyEvaluator, *, clock: Clock = utc_now
    ) -> None:
        self._judge = judge
        self._evaluator = evaluator
        self._clock = clock

    @override
    async def evaluate(self, interaction: Interaction, stage: Stage, cfg: ControlConfig) -> Verdict:
        if not self._judge.configured():
            return self._verdict(Decision.ALLOW, NOT_CONFIGURED)
        segments = _answer_segments(interaction)
        if not segments:
            return self._verdict(Decision.ALLOW, NO_CONTENT)
        if any(segment.kind not in _JUDGED_KINDS for segment in segments):
            return self._verdict(Decision.BLOCK, UNSCANNABLE, cfg)  # content the judge can't see
        judged = JudgedText.of(segments)
        quotes = await self._quotes(interaction, judged)
        if quotes is None:
            return self._verdict(Decision.BLOCK, UNAVAILABLE, cfg)
        return self._decide(cfg, quotes, *quote_spans(judged, quotes))

    async def _quotes(self, interaction: Interaction, judged: JudgedText) -> list[str] | None:
        """The judge's verdict as quotes ([] = in scope); None when it could not answer or
        there is no call scope to state (so nothing passes)."""
        instructions = self._instructions(interaction)
        if instructions is None:
            return None
        try:
            assessment = await self._judge.judge(
                control_id=self.id,
                instructions=instructions,
                content=judged.text,
                response_model=OutputAssessment,
            )
        except JudgeUnavailableError:
            return None
        return [violation.quote for violation in assessment.violations]

    def _decide(
        self, cfg: ControlConfig, quotes: list[str], spans: tuple[Span, ...], opaque: bool
    ) -> Verdict:
        if not quotes:
            return self._verdict(Decision.ALLOW, IN_SCOPE)
        if not spans:  # every quote was hallucinated or paraphrased: nothing to point at
            return self._verdict(Decision.ALLOW, QUOTES_UNMATCHED)
        risk = cfg.risk_delta or 0.0
        if cfg.mode is ControlMode.REDACT and not opaque:
            return self._verdict(Decision.REDACT, OUT_OF_SCOPE, cfg, risk=risk, spans=spans)
        return self._verdict(Decision.BLOCK, OUT_OF_SCOPE, cfg, risk=risk)

    def _instructions(self, interaction: Interaction) -> str | None:
        scope = current_scope()
        if scope is None:
            return None
        effective = self._evaluator.effective_scope(
            scope.snapshot, scope.principal, interaction.context, self._clock()
        )
        allow, deny = self._evaluator.agent_scope(scope.snapshot, scope.principal)

        def listed(permissions: tuple[str, ...]) -> str:
            return escape_delimiters(", ".join(permissions)) if permissions else "(nothing)"

        return INSTRUCTIONS.format(
            scope=listed(effective), agent_allow=listed(allow), agent_deny=listed(deny)
        )

    def _verdict(
        self,
        decision: Decision,
        reason_code: str,
        cfg: ControlConfig | None = None,
        *,
        risk: float = 0.0,
        spans: tuple[Span, ...] = (),
    ) -> Verdict:
        return Verdict(
            decision=decision,
            control_id=self.id,
            reason_code=reason_code,
            enforced=cfg is None or cfg.mode is not ControlMode.LOG_ONLY,
            risk_delta=risk,
            redactions=spans,
        )
