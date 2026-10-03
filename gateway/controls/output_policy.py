"""``output_policy``: does the model's answer reveal data outside the session's scope?

Post, LLM channel. The judge gets the session's effective scope (permission strings from the
evaluator, plus the agent's own allow and deny lists) as trusted instructions, and the
answer's text as untrusted data, and lists the passages that reveal out-of-scope data as
verbatim quotes. Each quote is mapped back to **exact** spans:

- the answer is every redactable text value of the completion (message content, reasoning,
  tool-call argument values); protocol fields (ids, roles, model names) and decoded base64
  are not sent;
- a quote counts only where it occurs verbatim inside one text value; every occurrence is a
  span (the same datum repeated is the same leak). A quote found nowhere is ignored: judges
  hallucinate and paraphrase, and a span must point at real text;
- ``redact`` masks the spans; ``block`` refuses the answer.

**Unavailable fails closed to block in both modes.** ``redact`` needs to know *which* spans
to mask; without a judge the gateway does not know, and masking the whole answer would be a
block with a misleading audit entry. No ``judges`` section: the control is off.

**Limits** (inherently fuzzy, kept conservative): the judge can only judge the scope it is
told and the text it sees, and it cannot know where a fact came from (a public fact that
happens to match a table is not a leak). A leak the judge reports with a paraphrase instead
of a verbatim quote is missed in both modes, because an unmatched quote is ignored. Data
split across two text values is not matched. Answers longer than
``judges.max_content_chars`` are not judged (unavailable → block): raise the limit or keep
answers short. Row-level security and
``authz`` remain the real boundary; this is a backstop for what the model says.
"""

from typing import ClassVar, Final, override

from pydantic import BaseModel, Field

from gateway.clock import Clock, utc_now
from gateway.controls.scope import current_scope
from gateway.controls.text import SegmentKind, TextExtractor, TextSegment
from gateway.core.envelope import Interaction, Span, Verdict
from gateway.core.interfaces import Control, ControlConfig
from gateway.core.types import ControlKind, ControlMode, Decision, Stage
from gateway.judges.client import JudgeClient, JudgeUnavailableError, escape_delimiters
from gateway.policy.evaluator import PolicyEvaluator

LABEL: Final = "OUT_OF_SCOPE"
IN_SCOPE: Final = "answer_in_scope"
OUT_OF_SCOPE: Final = "out_of_scope_data"
QUOTES_UNMATCHED: Final = "judge_quotes_unmatched"
UNAVAILABLE: Final = "judge_unavailable"
NO_CONTENT: Final = "no_content"
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
    quote: str = Field(max_length=4000)
    reason: str = Field(default="", max_length=500)


class OutputAssessment(BaseModel):
    violations: list[Violation] = Field(default_factory=list[Violation], max_length=MAX_VIOLATIONS)


def _answer_segments(interaction: Interaction) -> list[TextSegment]:
    """The text values of the completion a person would read as the answer."""
    return [
        segment
        for segment in TextExtractor().segments(interaction, Stage.POST)
        if segment.kind is SegmentKind.TEXT and segment.joinable and segment.text.strip()
    ]


def _occurrences(text: str, quote: str) -> list[tuple[int, int]]:
    """Every non-overlapping verbatim occurrence of ``quote`` in ``text``."""
    found: list[tuple[int, int]] = []
    start = text.find(quote)
    while start != -1:
        found.append((start, start + len(quote)))
        start = text.find(quote, start + len(quote))
    return found


def quote_spans(segments: list[TextSegment], quotes: list[str]) -> tuple[Span, ...]:
    """Spans over every verbatim occurrence of each quote in each segment (code points).

    The judge saw the answer with delimiter-like tags escaped, so a quote is also looked up
    with that escaping undone. Whitespace around a quote is ignored; an empty one matches
    nothing."""
    spans: list[Span] = []
    for raw in quotes:
        for quote in dict.fromkeys((raw.strip(), raw.strip().replace("&lt;", "<"))):
            if not quote:
                continue
            for segment in segments:
                for start, end in _occurrences(segment.text, quote):
                    spans.append(segment.span(start, end, LABEL))
    return tuple(dict.fromkeys(spans))


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
        quotes = await self._quotes(interaction, segments)
        if quotes is None:
            return self._verdict(Decision.BLOCK, UNAVAILABLE, cfg)
        return self._decide(cfg, quotes, quote_spans(segments, quotes))

    async def _quotes(
        self, interaction: Interaction, segments: list[TextSegment]
    ) -> list[str] | None:
        """The judge's verdict as quotes ([] = in scope); None when it could not answer or
        there is no call scope to state (so nothing passes)."""
        instructions = self._instructions(interaction)
        if instructions is None:
            return None
        try:
            assessment = await self._judge.judge(
                control_id=self.id,
                instructions=instructions,
                content=SEPARATOR.join(segment.text for segment in segments),
                response_model=OutputAssessment,
            )
        except JudgeUnavailableError:
            return None
        return [violation.quote for violation in assessment.violations]

    def _decide(self, cfg: ControlConfig, quotes: list[str], spans: tuple[Span, ...]) -> Verdict:
        if not quotes:
            return self._verdict(Decision.ALLOW, IN_SCOPE)
        if not spans:  # every quote was hallucinated or paraphrased: nothing to point at
            return self._verdict(Decision.ALLOW, QUOTES_UNMATCHED)
        risk = cfg.risk_delta or 0.0
        if cfg.mode is ControlMode.REDACT:
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
