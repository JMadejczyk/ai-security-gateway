"""``signatures``: match every call against the external attack-signature feed.

Runs pre and post on every channel. What each pattern type is matched against
(`gateway.feed.schema` defines the types):

- ``regex``: the free text of the payload (pre) or result (post), as `TextExtractor` yields it
  (messages and tool-call arguments for the LLM, arguments and tool results for MCP);
- ``mcp_tool``: tool names and descriptions: the tool a ``tools/call`` names and the tools an
  LLM request declares (pre), and every tool an MCP server advertises (`screen_listing`, which
  the MCP proxy calls on ``tools/list``: ``tools/list`` does not run through the pipeline);
- ``path_glob``: every string argument of an MCP ``tools/call`` (pre).

A match yields ``signature_match`` carrying the matched signature ids (ids only, never the
matched text) and the configured ``risk_delta``, enforced unless the mode is ``log_only``.
Severity is informational: every match adds the same ``risk_delta``, which keeps the risk
model one number per control, like every other control. A scan that times out fails closed
(``signature_scan_timeout``) and adds no risk, since nothing was detected.
"""

import logging
from collections.abc import Callable, Iterator, Mapping
from typing import Any, ClassVar, Final, cast, override

from gateway.controls.text import TextExtractor, string_leaves
from gateway.core.catalog import control_spec
from gateway.core.envelope import Interaction, Verdict
from gateway.core.interfaces import Control, ControlConfig
from gateway.core.types import Channel, ControlKind, ControlMode, Decision, Stage
from gateway.feed.schema import FILE_SCHEME, PatternType, ScanResult, SignatureFeed
from gateway.policy.loader import PolicySnapshot
from gateway.telemetry import record_signature_hits

logger = logging.getLogger(__name__)

SIGNATURE_MATCH: Final = "signature_match"
SCAN_TIMEOUT: Final = "signature_scan_timeout"
NO_MATCH: Final = "no_signature_match"
MAX_REASON_IDS: Final = 10


def _mapping(value: object) -> Mapping[str, Any]:
    return cast("Mapping[str, Any]", value) if isinstance(value, Mapping) else {}


def _items(value: object) -> list[Any]:
    return cast("list[Any]", value) if isinstance(value, list) else []


def declared_tools(interaction: Interaction) -> Iterator[str]:
    """Tool names and descriptions a request names: the MCP tool it calls, or the function
    tools an LLM request declares to the model."""
    payload = _mapping(interaction.payload)
    if interaction.channel is Channel.MCP:
        name = payload.get("name")
        if isinstance(name, str):
            yield name
        return
    for tool in _items(payload.get("tools")):
        function = _mapping(_mapping(tool).get("function"))
        for field in ("name", "description"):
            text = function.get(field)
            if isinstance(text, str):
                yield text


def path_arguments(payload: object) -> Iterator[str]:
    """Every string argument of a ``tools/call``, with a leading ``file://`` dropped."""
    for segment in string_leaves(_mapping(payload).get("arguments")):
        text = segment.text
        yield text[len(FILE_SCHEME) :] if text.lower().startswith(FILE_SCHEME) else text


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
        scan = self._feed().scan(interaction.channel)
        segments = self._extractor.segments(interaction, stage)
        scan.check(PatternType.REGEX, (segment.text for segment in segments))
        if stage is Stage.PRE:
            scan.check(PatternType.MCP_TOOL, declared_tools(interaction))
            if interaction.channel is Channel.MCP:
                scan.check(PatternType.PATH_GLOB, path_arguments(interaction.payload))
        return self._verdict(scan.result(), cfg)

    def screen_listing(self, tools: Mapping[str, str | None], snapshot: PolicySnapshot) -> set[str]:
        """Names of advertised MCP tools to hide from ``tools/list``.

        Every flagged tool is logged and counted; under ``log_only`` nothing is hidden. A tool
        whose scan times out is hidden too (fail closed).
        """
        feed, hidden = self._feed(), set[str]()
        for name, description in tools.items():
            scan = feed.scan(Channel.MCP)
            scan.check(PatternType.MCP_TOOL, (name, description or ""))
            result = scan.result()
            if result.matched or result.incomplete:
                hidden.add(name)
                record_signature_hits(result.ids)
                logger.warning(
                    "tools/list: an advertised tool (%r) is flagged by signatures %s",
                    name[:64],
                    ",".join(result.ids) or SCAN_TIMEOUT,
                )
        if snapshot.policy.resolved_control_mode(self.id) is ControlMode.LOG_ONLY:
            return set()
        return hidden

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
