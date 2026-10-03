"""Operator-approved MCP tool baselines: ``pins/<server>.json`` (SPEC ``tool_pinning``).

A pin file is a versioned `PinFile`, written by ``acl pin <server>`` from an operator-reviewed
``tools/list`` snapshot (``GET /admin/mcp/{server}/tools``). Each tool is pinned by name with
its description, input schema and annotations, plus a `tool_digest` over exactly those four
fields. ``tool_pinning`` compares every advertised tool with its baseline, and arguments are
validated against the pinned input schema.

`PinStore` is the one reader: it loads a pin file on demand and re-reads it only when it
changes on disk. A pin file that cannot be read or validated (unknown format, another server's
name, a digest that does not match its fields) fails closed: every call to that server is
refused with ``tool_pin_invalid`` until it is fixed.
"""

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Final, Literal, Self, cast

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from gateway.errors import RejectionError
from gateway.proxies.mcp import wire
from gateway.telemetry import canonical_json

logger = logging.getLogger(__name__)

MAX_PIN_BYTES: Final = 1024 * 1024
PIN_SCHEMA_VERSION: Final = 1

type Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
type ServerName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]


class PinFileError(RejectionError):
    def __init__(self) -> None:
        super().__init__("tool_pin_invalid", "the pinned tool baseline cannot be used")


def tool_digest(
    name: object, description: object, input_schema: object, annotations: object
) -> str:
    """SHA-256 of the canonical (sorted-key, compact) JSON of the four pinned fields; an absent
    field counts as ``null``. Takes raw JSON values, so an advertised tool is digested as sent."""
    document = {
        "name": name,
        "description": description,
        "inputSchema": input_schema,
        "annotations": annotations,
    }
    return hashlib.sha256(canonical_json(document)).hexdigest()


def advertised_digest(tool: wire.ToolDefinition) -> str:
    """The digest of a tool exactly as the upstream advertised it."""
    extra = tool.model_extra or {}
    return tool_digest(
        tool.name, extra.get("description"), tool.input_schema, extra.get("annotations")
    )


class _PinModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True, allow_inf_nan=False
    )


class ToolBaseline(_PinModel):
    """One approved tool: the four pinned fields and their digest."""

    name: str = Field(min_length=1)
    description: str | None = None
    input_schema: dict[str, Any] = Field(alias="inputSchema")
    annotations: dict[str, Any] | None = None
    digest: Digest

    @model_validator(mode="after")
    def _digest_matches(self) -> Self:
        if self.digest != self.computed_digest():
            msg = f"tool {self.name!r}: digest does not match its pinned fields"
            raise ValueError(msg)
        return self

    def computed_digest(self) -> str:
        return tool_digest(self.name, self.description, self.input_schema, self.annotations)

    @classmethod
    def of(cls, tool: wire.ToolDefinition) -> Self:
        """The baseline of an advertised tool; `ValidationError` when its description or
        annotations are not a string / an object."""
        extra = tool.model_extra or {}
        return cls.model_validate(
            {
                "name": tool.name,
                "description": extra.get("description"),
                "inputSchema": tool.input_schema,
                "annotations": extra.get("annotations"),
                "digest": advertised_digest(tool),
            }
        )

    def as_definition(self) -> wire.ToolDefinition:
        """The tool as the gateway lists it: the approved fields only. An upstream's ``title``,
        ``outputSchema`` or ``_meta`` were never reviewed and are never relayed."""
        fields: dict[str, Any] = {"name": self.name, "inputSchema": self.input_schema}
        if self.description is not None:
            fields["description"] = self.description
        if self.annotations is not None:
            fields["annotations"] = self.annotations
        return wire.ToolDefinition.model_validate(fields)


class PinFile(_PinModel):
    """``pins/<server>.json``: the approved baseline of every tool one server advertises."""

    schema_version: Literal[1] = PIN_SCHEMA_VERSION
    server: ServerName
    captured_at: AwareDatetime
    protocol_version: str = Field(min_length=1)
    tools: tuple[ToolBaseline, ...]

    @model_validator(mode="after")
    def _unique_names(self) -> Self:
        names = [tool.name for tool in self.tools]
        if len(set(names)) != len(names):
            msg = "a pin file lists a tool name twice"
            raise ValueError(msg)
        return self

    @classmethod
    def capture(
        cls, server: str, tools: Sequence[wire.ToolDefinition], captured_at: datetime
    ) -> Self:
        """A candidate baseline from an advertised listing, tools sorted by name.

        Raises `ValidationError` for duplicate names or malformed fields: an ambiguous
        listing cannot be approved."""
        baselines = sorted((ToolBaseline.of(tool) for tool in tools), key=lambda t: t.name)
        return cls(
            server=server,
            captured_at=captured_at,
            protocol_version=wire.PROTOCOL_VERSION,
            tools=tuple(baselines),
        )

    def tool(self, name: str) -> ToolBaseline | None:
        return next((tool for tool in self.tools if tool.name == name), None)

    @property
    def revision(self) -> str:
        """Identifies the approved baseline: SHA-256 over every tool's name and digest."""
        entries = sorted((tool.name, tool.digest) for tool in self.tools)
        return hashlib.sha256(canonical_json(entries)).hexdigest()

    @property
    def schemas(self) -> wire.ToolSchemas:
        return MappingProxyType({tool.name: tool.input_schema for tool in self.tools})

    def to_json(self) -> str:
        return self.model_dump_json(by_alias=True, indent=2) + "\n"


# --------------------------------------------------------------------------- diff


class _Missing:
    """A JSON member present on one side of a `FieldChange` only."""

    def __repr__(self) -> str:
        return "(absent)"


MISSING: Final = _Missing()


@dataclass(frozen=True, slots=True)
class FieldChange:
    pointer: str  # JSON pointer into the tool, e.g. "/inputSchema/properties/sql/maxLength"
    before: object  # MISSING when the member was added
    after: object  # MISSING when the member was removed


@dataclass(frozen=True, slots=True)
class ToolChange:
    name: str
    changes: tuple[FieldChange, ...]


@dataclass(frozen=True, slots=True)
class PinDiff:
    """What approving ``new`` changes relative to ``old`` (None: no pin file yet)."""

    added: tuple[ToolBaseline, ...]
    removed: tuple[ToolBaseline, ...]
    changed: tuple[ToolChange, ...]
    protocol: tuple[str, str] | None = None  # (before, after) when the version differs

    @property
    def empty(self) -> bool:
        return not (self.added or self.removed or self.changed or self.protocol)

    @classmethod
    def between(cls, old: PinFile | None, new: PinFile) -> Self:
        before = {tool.name: tool for tool in old.tools} if old is not None else {}
        after = {tool.name: tool for tool in new.tools}
        changed = tuple(
            ToolChange(name, tuple(json_changes(_fields(before[name]), _fields(tool))))
            for name, tool in sorted(after.items())
            if name in before and before[name].digest != tool.digest
        )
        protocol = (
            (old.protocol_version, new.protocol_version)
            if old is not None and old.protocol_version != new.protocol_version
            else None
        )
        return cls(
            added=tuple(tool for name, tool in sorted(after.items()) if name not in before),
            removed=tuple(tool for name, tool in sorted(before.items()) if name not in after),
            changed=changed,
            protocol=protocol,
        )


def _fields(tool: ToolBaseline) -> dict[str, object]:
    return tool.model_dump(mode="json", by_alias=True, exclude={"digest"})


def _escape(key: str) -> str:
    return key.replace("~", "~0").replace("/", "~1")


def _members(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    return {str(k): v for k, v in cast("dict[object, object]", value).items()}


def json_changes(before: object, after: object, pointer: str = "") -> list[FieldChange]:
    """Leaf-level differences between two JSON values, addressed by JSON pointer."""
    left, right = _members(before), _members(after)
    if left is not None and right is not None:
        changes: list[FieldChange] = []
        for key in sorted(left.keys() | right.keys()):
            changes += json_changes(
                left.get(key, MISSING), right.get(key, MISSING), f"{pointer}/{_escape(key)}"
            )
        return changes
    if before is MISSING or after is MISSING or canonical_json(before) != canonical_json(after):
        return [FieldChange(pointer or "/", before, after)]
    return []


# ------------------------------------------------------------------------- reader


@dataclass(frozen=True, slots=True)
class _Cached:
    mtime_ns: int
    size: int
    pin: PinFile


class PinStore:
    """Reads pin files on demand and re-reads one only when it changes on disk."""

    def __init__(self, pins_dir: Path) -> None:
        self._dir = pins_dir
        self._cache: dict[str, _Cached] = {}

    def path(self, server: str) -> Path:
        return self._dir / f"{server}.json"

    def lookup(self, server: str) -> PinFile | None:
        """The pin file of ``server``; None when it has none. Raises `PinFileError`."""
        path = self.path(server)
        try:
            stat = path.stat()
        except FileNotFoundError:
            self._cache.pop(server, None)
            return None
        except OSError:
            logger.exception("cannot stat pin file for MCP server %s", server)
            raise PinFileError from None
        cached = self._cache.get(server)
        if cached is not None and (cached.mtime_ns, cached.size) == (
            stat.st_mtime_ns,
            stat.st_size,
        ):
            return cached.pin
        pin = self._load(path, server, stat.st_size)
        self._cache[server] = _Cached(stat.st_mtime_ns, stat.st_size, pin)
        return pin

    @staticmethod
    def _load(path: Path, server: str, size: int) -> PinFile:
        if size > MAX_PIN_BYTES:
            logger.error("pin file for MCP server %s exceeds %d bytes", server, MAX_PIN_BYTES)
            raise PinFileError
        try:
            pin = PinFile.model_validate_json(path.read_bytes())
        except (OSError, ValidationError):
            logger.exception("invalid pin file for MCP server %s (re-run `acl pin`)", server)
            raise PinFileError from None
        if pin.server != server:
            logger.error("pin file %s.json pins another server", server)
            raise PinFileError
        return pin
