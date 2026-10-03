"""``tool_poisoning``: the injection classifier on MCP tool definitions.

A poisoned tool hides instructions for the model in what the server says about the tool
(its description, a parameter's description, a title, a default), the text an agent reads
when it decides which tool to call. Two checks, both on the same text per tool
(`definition_text`) and the same `ClassifierRunner` as ``prompt_injection``:

- **at registration** (`screen_listing`, one of the MCP proxy's ``tools/list`` screens): every
  advertised definition, after ``tool_pinning`` verified the listing, is classified under one
  deadline (`LISTING_BUDGET_S`); a tool scoring at or above ``threshold`` is hidden. Like the
  ``signatures`` screen it fails closed: a tool that could not be classified in time, one past
  `MAX_LISTED_TOOLS`, or every tool when the classifier cannot run, is hidden too. A name
  listed twice is hidden when any of its entries is poisoned.
- **at ``tools/call``** (`evaluate`, pre): the called tool's definition from the caller's own
  verified listing (``tool_pinning``'s `current_listing`), so a poisoned tool that was hidden
  and is then called by name is blocked with ``tool_poisoning_detected``. A call without a
  published listing fails closed (``tool_definition_unavailable``); a name that listing does
  not carry is allowed here (``tool_not_advertised``): ``tool_pinning`` and the adapter refuse it.

There is no judge tier: definitions are short, static and reviewed by the operator when the
server is pinned, and a judge call per tool would put seconds of LLM latency on ``tools/list``.
Scores are cached by text, so a server's definitions are classified once per process.
"""

import asyncio
import logging
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, ClassVar, Final, cast, override

from gateway.controls.prompt_injection import Finding, finding_verdict, score_bucket
from gateway.controls.tool_pinning import current_listing
from gateway.core.envelope import Interaction, Verdict
from gateway.core.interfaces import Control, ControlConfig
from gateway.core.types import Channel, ControlKind, Decision, Stage
from gateway.injection.classifier import ClassifierRunner, ClassifierUnavailableError
from gateway.policy.loader import PolicySnapshot
from gateway.policy.schema import ToolPoisoningConfig
from gateway.proxies.mcp import wire
from gateway.telemetry import record_verdicts

logger = logging.getLogger(__name__)

DETECTED: Final = "tool_poisoning_detected"
CLEAN: Final = "no_tool_poisoning"
NOT_ADVERTISED: Final = "tool_not_advertised"
DEFINITION_UNAVAILABLE: Final = "tool_definition_unavailable"
SCREEN_TIMEOUT: Final = "tool_screen_timeout"
CLASSIFIER_UNAVAILABLE: Final = "classifier_unavailable"
LISTING_TOO_LARGE: Final = "tool_listing_too_large"

LISTING_BUDGET_S: Final = 10.0  # one deadline for classifying a whole listing
MAX_LISTED_TOOLS: Final = 256  # tools screened per listing; any beyond are hidden
# JSON Schema keywords whose string values are identifiers or grammar, not prose the model
# reads as advice. Skipped only where they are keywords: never inside a data value.
_IDENTIFIER_KEYS: Final = frozenset(
    {"type", "format", "$schema", "$id", "$ref", "pattern", "required", "mimeType", "uri"}
)
# JSON Schema keywords whose values are instance data: every string inside them, object keys
# included, is text the model may read (``"default": {"type": "ignore previous..."}``).
_DATA_KEYS: Final = frozenset({"default", "examples", "enum", "const"})


def _data_strings(value: object) -> Iterator[str]:
    """Every string in a JSON value: values and object keys, at any depth."""
    if isinstance(value, str):
        if value:
            yield value
    elif isinstance(value, Mapping):
        for child_key, child in cast("Mapping[str, Any]", value).items():
            yield child_key
            yield from _data_strings(child)
    elif isinstance(value, list):
        for child in cast("list[Any]", value):
            yield from _data_strings(child)


def _strings(value: object, key: str | None = None) -> Iterator[str]:
    if key in _DATA_KEYS:
        yield from _data_strings(value)
    elif isinstance(value, str):
        if value and key not in _IDENTIFIER_KEYS:
            yield value
    elif isinstance(value, Mapping):
        for child_key, child in cast("Mapping[str, Any]", value).items():
            if key == "properties":  # parameter names are text the model reads too
                yield child_key
            yield from _strings(child, child_key)
    elif isinstance(value, list):
        for child in cast("list[Any]", value):
            yield from _strings(child, key)


def definition_text(tool: wire.ToolDefinition) -> str:
    """Everything a tool definition tells the model, one line per string: its name, the
    description, titles, every schema description, parameter names, and every string inside a
    ``default``, ``examples``, ``enum`` or ``const`` value, object keys included."""
    return "\n".join(dict.fromkeys(_strings(tool.as_wire())))


class ToolPoisoningControl(Control):
    id: ClassVar[str] = "tool_poisoning"
    stages: ClassVar[frozenset[Stage]] = frozenset({Stage.PRE})
    kind: ClassVar[ControlKind] = ControlKind.SEMANTIC

    def __init__(
        self, runner: ClassifierRunner, *, listing_budget_s: float = LISTING_BUDGET_S
    ) -> None:
        self._runner = runner
        self._budget_s = listing_budget_s

    @override
    async def evaluate(self, interaction: Interaction, stage: Stage, cfg: ControlConfig) -> Verdict:
        config = _typed(cfg)
        finding = await self._check_call(interaction, config)
        if not finding.clean:
            logger.warning(
                "tool_poisoning %s on tools/call: %s", finding.reason_code, finding.reason
            )
        return finding_verdict(self.id, finding, config)

    async def _check_call(self, interaction: Interaction, config: ToolPoisoningConfig) -> Finding:
        payload: object = interaction.payload
        if interaction.channel is not Channel.MCP or not isinstance(payload, dict):
            return Finding.passed(CLEAN)
        name = cast("dict[str, Any]", payload).get("name")
        listing = current_listing()
        if listing is None:
            return Finding.refusal(DEFINITION_UNAVAILABLE, "no verified listing for this call")
        entries = [tool for tool in listing.definitions if tool.name == name]
        if not entries:
            return Finding.passed(NOT_ADVERTISED)
        try:
            scores = await self._runner.scores([definition_text(tool) for tool in entries])
        except ClassifierUnavailableError:
            return Finding.refusal(CLASSIFIER_UNAVAILABLE, "the injection classifier is disabled")
        top = max(score.score for score in scores)
        if top >= config.threshold:
            return Finding.detection(DETECTED, f"classifier score {score_bucket(top)}")
        return Finding.passed(CLEAN)

    async def screen_listing(
        self, server: str, tools: Sequence[wire.ToolDefinition], snapshot: PolicySnapshot
    ) -> set[str]:
        """Names of advertised tools to hide from ``tools/list`` (see the module docstring)."""
        config = _typed(snapshot.policy.control_config(self.id))
        flagged = await self._flag(tools, config.threshold)
        for name, (reason_code, reason) in flagged.items():
            record_verdicts(
                [Verdict(decision=Decision.BLOCK, control_id=self.id, reason_code=reason_code)]
            )
            logger.warning(
                "tools/list %s: an advertised tool (%r) is hidden: %s %s",
                server,
                name[:64],
                reason_code,
                reason,
            )
        return set(flagged)

    async def _flag(
        self, tools: Sequence[wire.ToolDefinition], threshold: float
    ) -> dict[str, tuple[str, str]]:
        screened = list(tools[:MAX_LISTED_TOOLS])
        # A name with any entry past the cap is hidden, even if earlier entries of it are
        # clean: the agent would be shown the unclassified one too.
        flagged = {tool.name: (LISTING_TOO_LARGE, "") for tool in tools[MAX_LISTED_TOOLS:]}
        try:
            async with asyncio.timeout(self._budget_s):
                scores = await self._runner.scores([definition_text(t) for t in screened])
        except TimeoutError:
            return flagged | {t.name: (SCREEN_TIMEOUT, "") for t in screened}
        except ClassifierUnavailableError:
            return flagged | {t.name: (CLASSIFIER_UNAVAILABLE, "") for t in screened}
        best: defaultdict[str, float] = defaultdict(float)
        for tool, score in zip(screened, scores, strict=True):
            best[tool.name] = max(best[tool.name], score.score)
        for name, score in best.items():
            if score >= threshold:
                flagged[name] = (DETECTED, f"classifier score {score_bucket(score)}")
        return flagged


def _typed(cfg: ControlConfig) -> ToolPoisoningConfig:
    if isinstance(cfg, ToolPoisoningConfig):
        return cfg
    return ToolPoisoningConfig(mode=cfg.mode, risk_delta=cfg.risk_delta)
