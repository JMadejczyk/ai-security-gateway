"""``tool_pinning``: every MCP tool must look exactly like its operator-approved baseline.

SPEC "Control catalog". A server's baseline is its pin file (`gateway.proxies.mcp.pins`).
Each advertised tool is compared, by digest over name, description, input schema and
annotations, with its baseline:

- equal: the tool is usable, and ``tools/list`` shows the *pinned* definition (only reviewed
  fields reach the agent, never an upstream's extra ``title``, ``outputSchema`` or ``_meta``);
- different, advertised twice, or pinned but no longer advertised: ``tool_pin_mismatch``
  (a rug pull) — hidden from ``tools/list`` and blocked on ``tools/call``;
- advertised but not in the baseline: ``tool_not_pinned``, hidden and blocked.

A tool stays blocked for as long as it differs from the baseline, i.e. until an operator
reviews the change and re-pins (``acl pin <server> --write``). Pinning does not prove what the
server's implementation does; it proves the agent only ever sees and calls the approved surface.

Servers without a pin file: ``upstreams.mcp.<server>.require_pin`` (default true) blocks
every tool as ``tool_not_pinned``; ``require_pin: false`` opts a server out explicitly, and its
tools are compared with nothing (the upstream's first listing supplies argument schemas).

Which listing a call is checked against: the MCP proxy keeps each upstream session's latest
``tools/list`` (fetched on demand when the agent calls a tool by name without listing), verifies
it with `ToolPinningControl.verify` and publishes the result for the call (`listing_scope`).
The listing is not re-fetched on every call: a description only matters once it reaches the
model, which happens only through ``tools/list``, and every ``tools/list`` is fetched fresh
and verified; arguments are always validated against the pinned schema.
"""

import logging
from collections import Counter
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any, ClassVar, Final, cast, override

from gateway.core.envelope import Interaction, Verdict
from gateway.core.interfaces import Control, ControlConfig
from gateway.core.types import ControlKind, Decision, Stage
from gateway.policy.loader import PolicySnapshot
from gateway.policy.schema import McpServer
from gateway.proxies.mcp import wire
from gateway.proxies.mcp.pins import PinFile, PinStore, advertised_digest
from gateway.proxies.mcp.upstream import MCPUpstream
from gateway.telemetry import record_verdicts

logger = logging.getLogger(__name__)

PIN_UNVERIFIED: Final = "tool_pin_unverified"


class PinStatus(StrEnum):
    PINNED = "tool_pinned"  # matches its baseline
    UNPINNED_ALLOWED = "tool_unpinned_allowed"  # no pin file, `require_pin: false`
    MISMATCH = "tool_pin_mismatch"
    NOT_PINNED = "tool_not_pinned"

    @property
    def usable(self) -> bool:
        return self in {PinStatus.PINNED, PinStatus.UNPINNED_ALLOWED}


@dataclass(frozen=True, slots=True)
class PinnedListing:
    """One upstream listing after verification against the server's pin file."""

    server: str
    pin: PinFile | None
    statuses: Mapping[str, PinStatus]  # every advertised or pinned tool name
    unknown: PinStatus  # status of a name neither advertised nor pinned
    definitions: tuple[wire.ToolDefinition, ...]  # usable tools as listed, advertised order
    schemas: wire.ToolSchemas  # argument schemas for the adapter

    def status(self, name: str) -> PinStatus:
        return self.statuses.get(name, self.unknown)

    def definition(self, name: str) -> wire.ToolDefinition | None:
        """The verified definition of a usable tool, as the agent was shown it."""
        return next((tool for tool in self.definitions if tool.name == name), None)

    @property
    def hidden(self) -> dict[str, PinStatus]:
        return {name: status for name, status in self.statuses.items() if not status.usable}


_LISTING: ContextVar[PinnedListing | None] = ContextVar("acl_pinned_listing", default=None)


@contextmanager
def listing_scope(listing: PinnedListing) -> Generator[PinnedListing]:
    """Publish the verified listing of the calling upstream session for one ``tools/call``."""
    token = _LISTING.set(listing)
    try:
        yield listing
    finally:
        _LISTING.reset(token)


def current_listing() -> PinnedListing | None:
    return _LISTING.get()


def _first_schemas(tools: Sequence[wire.ToolDefinition]) -> dict[str, dict[str, Any]]:
    schemas: dict[str, dict[str, Any]] = {}
    for tool in tools:
        schemas.setdefault(tool.name, tool.input_schema)
    return schemas


class ToolPinningControl(Control):
    id: ClassVar[str] = "tool_pinning"
    stages: ClassVar[frozenset[Stage]] = frozenset({Stage.PRE})
    kind: ClassVar[ControlKind] = ControlKind.DETERMINISTIC

    def __init__(self, pins: PinStore) -> None:
        self._pins = pins

    # ------------------------------------------------------------------ verification

    def verify(
        self, server: str, config: McpServer, tools: Sequence[wire.ToolDefinition]
    ) -> PinnedListing:
        """Compare one advertised listing with the server's baseline. Raises `PinFileError`."""
        pin = self._pins.lookup(server)
        if pin is None:
            status = PinStatus.NOT_PINNED if config.require_pin else PinStatus.UNPINNED_ALLOWED
            return PinnedListing(
                server=server,
                pin=None,
                statuses=MappingProxyType(dict.fromkeys((t.name for t in tools), status)),
                unknown=status,
                definitions=tuple(tools) if status.usable else (),
                schemas=MappingProxyType(_first_schemas(tools)),
            )
        counts = Counter(tool.name for tool in tools)
        statuses: dict[str, PinStatus] = {}
        for tool in tools:
            baseline = pin.tool(tool.name)
            if baseline is None:
                status = PinStatus.NOT_PINNED
            elif counts[tool.name] > 1 or advertised_digest(tool) != baseline.digest:
                status = PinStatus.MISMATCH  # a second entry must not stand in for the first
            else:
                status = PinStatus.PINNED
            if (seen := statuses.get(tool.name)) is None or seen.usable:
                statuses[tool.name] = status
        for baseline in pin.tools:
            statuses.setdefault(baseline.name, PinStatus.MISMATCH)  # no longer advertised
        definitions = tuple(
            baseline.as_definition()
            for name in dict.fromkeys(tool.name for tool in tools)
            if statuses[name] is PinStatus.PINNED and (baseline := pin.tool(name)) is not None
        )
        # Pinned schemas win; an unpinned tool keeps its own so the call reaches this control
        # and is refused as `tool_not_pinned` rather than as an unknown tool.
        schemas = _first_schemas(tools) | dict(pin.schemas)
        return PinnedListing(
            server=server,
            pin=pin,
            statuses=MappingProxyType(statuses),
            unknown=PinStatus.NOT_PINNED,
            definitions=definitions,
            schemas=MappingProxyType(schemas),
        )

    def screen_listing(
        self, server: str, config: McpServer, tools: Sequence[wire.ToolDefinition]
    ) -> PinnedListing:
        """`verify` for ``tools/list``: every hidden tool is logged and counted."""
        listing = self.verify(server, config, tools)
        for name, status in listing.hidden.items():
            record_verdicts(
                [Verdict(decision=Decision.BLOCK, control_id=self.id, reason_code=status.value)]
            )
            logger.warning(
                "tools/list: MCP server %s tool %r is hidden: %s", server, name[:64], status.value
            )
        return listing

    async def for_call(
        self, server: str, config: McpServer, upstream: MCPUpstream, snapshot: PolicySnapshot
    ) -> PinnedListing:
        """The verified latest listing of the caller's upstream session, listing it first if
        the agent never did. Raises `PinFileError` or `UpstreamError`."""
        listing = self.verify(server, config, await upstream.latest_listing(snapshot))
        if listing.pin is None and listing.unknown.usable:  # opted out: the first listing's
            listing = replace(listing, schemas=await upstream.advertised_schemas(snapshot))
        return listing

    # ------------------------------------------------------------------------ control

    @override
    async def evaluate(self, interaction: Interaction, stage: Stage, cfg: ControlConfig) -> Verdict:
        payload: object = interaction.payload
        name = (
            cast("Mapping[str, Any]", payload).get("name") if isinstance(payload, Mapping) else None
        )
        listing = current_listing()
        if listing is None or listing.server != interaction.server or not isinstance(name, str):
            return self._block(PIN_UNVERIFIED, "no verified tool listing for this call", cfg)
        status = listing.status(name)
        if status.usable:
            return Verdict(decision=Decision.ALLOW, control_id=self.id, reason_code=status.value)
        return self._block(status.value, "the tool differs from its approved baseline", cfg)

    def _block(self, reason_code: str, reason: str, cfg: ControlConfig) -> Verdict:
        return Verdict(
            decision=Decision.BLOCK,
            control_id=self.id,
            reason_code=reason_code,
            reason=reason,
            risk_delta=cfg.risk_delta or 0.0,
        )
