"""``signatures``: match every call against the external attack-signature feed.

Runs pre and post on every channel. What each pattern type is matched against
(`gateway.feed.schema` defines the types):

- ``regex``: the free text of the payload (pre) or result (post), as `TextExtractor` yields it
  (messages and tool-call arguments for the LLM, arguments and tool results for MCP);
- ``mcp_tool``: tool names and descriptions: the tools an LLM request declares (pre), every
  tool an MCP server advertises (`screen_listing`, which the MCP proxy calls on
  ``tools/list``: ``tools/list`` does not run through the pipeline), and, at ``tools/call``,
  the called tool's name plus its definition from the caller's own verified listing
  (``tool_pinning``'s `current_listing`, as ``tool_poisoning`` does). A tool hidden from the
  listing for its description is therefore still refused when called by name. A call without
  a published listing fails closed (``tool_definition_unavailable``); a name that listing does
  not carry is checked by name only, and ``tool_pinning`` and the adapter refuse it.
  A listing is screened off the event loop under one deadline; a tool that could not be
  scanned in time, one past `MAX_LISTED_TOOLS`, or one whose name appears twice (a second,
  benign entry must not stand in for a poisoned one) is hidden like a matching one;
- ``path_glob``: every string argument of an MCP ``tools/call`` (pre).

A match yields ``signature_match`` carrying the matched signature ids (ids only, never the
matched text) and the configured ``risk_delta``, enforced unless the mode is ``log_only``.
Severity is informational: every match adds the same ``risk_delta``, which keeps the risk
model one number per control, like every other control. A scan that times out fails closed
(``signature_scan_timeout``) and adds no risk, since nothing was detected.
"""

import asyncio
import logging
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, ClassVar, Final, cast, override

from gateway.controls.scope import current_scope
from gateway.controls.text import TextExtractor, string_leaves
from gateway.controls.tool_pinning import current_listing
from gateway.core.catalog import control_spec
from gateway.core.envelope import Interaction, Verdict
from gateway.core.interfaces import Control, ControlConfig
from gateway.core.types import Channel, ControlKind, ControlMode, Decision, Stage
from gateway.feed.schema import (
    FILE_SCHEME,
    SCAN_BUDGET_S,
    PatternType,
    ScanResult,
    SignatureFeed,
)
from gateway.policy.loader import PolicySnapshot
from gateway.telemetry import record_signature_hits, record_verdicts

logger = logging.getLogger(__name__)

SIGNATURE_MATCH: Final = "signature_match"
SCAN_TIMEOUT: Final = "signature_scan_timeout"
NO_MATCH: Final = "no_signature_match"
DUPLICATE_TOOL: Final = "duplicate_tool_name"
DEFINITION_UNAVAILABLE: Final = "tool_definition_unavailable"
LISTING_TOO_LARGE: Final = "tool_listing_too_large"
MAX_REASON_IDS: Final = 10
MAX_LISTED_TOOLS: Final = 256  # tools screened per listing; any beyond are hidden
LISTING_BUDGET_S: Final = SCAN_BUDGET_S  # one deadline for the whole listing

type AdvertisedTools = Sequence[tuple[str, str | None]]  # (name, description), as listed


def _mapping(value: object) -> Mapping[str, Any]:
    return cast("Mapping[str, Any]", value) if isinstance(value, Mapping) else {}


def _items(value: object) -> list[Any]:
    return cast("list[Any]", value) if isinstance(value, list) else []


def declared_tools(interaction: Interaction) -> Iterator[str]:
    """Names and descriptions of the function tools an LLM request declares to the model."""
    payload = _mapping(interaction.payload)
    for tool in _items(payload.get("tools")):
        function = _mapping(_mapping(tool).get("function"))
        for field in ("name", "description"):
            text = function.get(field)
            if isinstance(text, str):
                yield text


def called_tool(interaction: Interaction) -> tuple[list[str], bool]:
    """Name and description(s) of the MCP tool a ``tools/call`` names, from the caller's
    verified listing; the flag is False when no listing was published for the call."""
    name = _mapping(interaction.payload).get("name")
    texts = [name] if isinstance(name, str) else []
    listing = current_listing()
    if listing is None:
        return texts, False
    for tool in listing.definitions:  # every entry of that name: duplicates included
        description = (tool.model_extra or {}).get("description")
        if tool.name == name and isinstance(description, str):
            texts.append(description)
    return texts, True


def path_arguments(payload: object) -> Iterator[str]:
    """Every string argument of a ``tools/call``, with a leading ``file://`` dropped."""
    for segment in string_leaves(_mapping(payload).get("arguments")):
        text = segment.text
        yield text[len(FILE_SCHEME) :] if text.lower().startswith(FILE_SCHEME) else text


def flag_listing(
    feed: SignatureFeed, tools: AdvertisedTools, *, budget_s: float = LISTING_BUDGET_S
) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Tools to hide: name -> (reason code, matched signature ids). Blocking; run off-loop.

    Every entry is scanned (duplicates included) under one deadline for the whole listing.
    """
    deadline = time.monotonic() + budget_s
    occurrences = Counter(name for name, _ in tools)
    flagged: dict[str, tuple[str, tuple[str, ...]]] = {
        name: (DUPLICATE_TOOL, ()) for name, seen in occurrences.items() if seen > 1
    }
    for index, (name, description) in enumerate(tools):
        if index >= MAX_LISTED_TOOLS:
            flagged.setdefault(name, (LISTING_TOO_LARGE, ()))
            continue
        scan = feed.scan(Channel.MCP, deadline=deadline)
        scan.check(PatternType.MCP_TOOL, (name, description or ""))
        result = scan.result()
        if result.matched:
            flagged[name] = (SIGNATURE_MATCH, result.ids)
        elif result.incomplete:
            flagged.setdefault(name, (SCAN_TIMEOUT, ()))
    return flagged


class SignaturesControl(Control):
    id: ClassVar[str] = "signatures"
    stages: ClassVar[frozenset[Stage]] = frozenset({Stage.PRE, Stage.POST})
    kind: ClassVar[ControlKind] = ControlKind.DETERMINISTIC

    def __init__(
        self, feed: Callable[[], SignatureFeed], extractor: TextExtractor | None = None
    ) -> None:
        self._feed = feed
        self._extractor = extractor or TextExtractor()

    @override
    async def evaluate(self, interaction: Interaction, stage: Stage, cfg: ControlConfig) -> Verdict:
        scope = current_scope()  # the call's pinned feed: pre and post see the same instance
        feed = scope.feed if scope is not None and scope.feed is not None else self._feed()
        scan = feed.scan(interaction.channel)
        segments = self._extractor.segments(interaction, stage)
        scan.check(PatternType.REGEX, (segment.text for segment in segments))
        listed = True
        if stage is Stage.PRE and interaction.channel is Channel.MCP:
            tool_texts, listed = called_tool(interaction)
            scan.check(PatternType.MCP_TOOL, tool_texts)
            scan.check(PatternType.PATH_GLOB, path_arguments(interaction.payload))
        elif stage is Stage.PRE:
            scan.check(PatternType.MCP_TOOL, declared_tools(interaction))
        result = scan.result()
        if not listed and not result.matched:
            return Verdict(
                decision=Decision.BLOCK,
                control_id=self.id,
                reason_code=DEFINITION_UNAVAILABLE,
                reason="no verified listing for this call",
                enforced=cfg.mode is not ControlMode.LOG_ONLY,
            )
        return self._verdict(result, cfg)

    async def screen_listing(self, tools: AdvertisedTools, snapshot: PolicySnapshot) -> set[str]:
        """Names of advertised MCP tools to hide from ``tools/list`` (see the module docstring).

        Every flagged tool is logged and counted; under ``log_only`` nothing is hidden.
        """
        flagged = await asyncio.to_thread(flag_listing, self._feed(), list(tools))
        for name, (reason_code, ids) in flagged.items():
            record_signature_hits(ids)
            record_verdicts(
                [Verdict(decision=Decision.BLOCK, control_id=self.id, reason_code=reason_code)]
            )
            logger.warning(
                "tools/list: an advertised tool (%r) is flagged: %s %s",
                name[:64],
                reason_code,
                ",".join(ids),
            )
        if snapshot.policy.resolved_control_mode(self.id) is ControlMode.LOG_ONLY:
            return set()
        return set(flagged)

    def _verdict(self, result: ScanResult, cfg: ControlConfig) -> Verdict:
        enforced = cfg.mode is not ControlMode.LOG_ONLY
        if result.matched:
            ids = result.ids
            record_signature_hits(ids)
            logger.info("signature_match ids=%s", ",".join(ids))
            risk_delta = cfg.risk_delta
            return Verdict(
                decision=Decision.BLOCK,
                control_id=self.id,
                reason_code=SIGNATURE_MATCH,
                reason=f"matched: {', '.join(ids[:MAX_REASON_IDS])}",
                enforced=enforced,
                risk_delta=(
                    risk_delta
                    if risk_delta is not None
                    else control_spec(self.id).default_risk_delta
                ),
            )
        if result.incomplete:
            return Verdict(
                decision=Decision.BLOCK,
                control_id=self.id,
                reason_code=SCAN_TIMEOUT,
                reason="a signature scan ran out of time",
                enforced=enforced,
            )
        return Verdict(decision=Decision.ALLOW, control_id=self.id, reason_code=NO_MATCH)
