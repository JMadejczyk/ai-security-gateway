"""Operator-pinned tool schemas: ``pins/<server>.json`` (SPEC "GenericMCPAdapter").

A pin file is an operator-reviewed ``tools/list`` result (``{"tools": [...]}``), written by
``acl pin <server>``. When it exists, its input schemas are the ones arguments are validated
against; otherwise the schema the upstream advertised first is used. A pin file that cannot be
read or validated fails closed: every call to that server is refused until it is fixed.
``tool_pinning`` (stage 3) will additionally compare names, descriptions and annotations.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from pydantic import ValidationError

from gateway.errors import RejectionError
from gateway.proxies.mcp.wire import ListToolsResult, ToolSchemas

logger = logging.getLogger(__name__)

MAX_PIN_BYTES: Final = 1024 * 1024


class PinFileError(RejectionError):
    def __init__(self) -> None:
        super().__init__("tool_pin_invalid", "the pinned tool schemas cannot be used")


@dataclass(frozen=True, slots=True)
class _Cached:
    mtime_ns: int
    size: int
    schemas: ToolSchemas


class PinnedSchemas:
    """Reads pin files on demand and re-reads one only when it changes on disk."""

    def __init__(self, pins_dir: Path) -> None:
        self._dir = pins_dir
        self._cache: dict[str, _Cached] = {}

    def lookup(self, server: str) -> ToolSchemas | None:
        """The pinned schemas of ``server``; None when it has no pin file."""
        path = self._dir / f"{server}.json"
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
            return cached.schemas
        schemas = self._load(path, server, stat.st_size)
        self._cache[server] = _Cached(stat.st_mtime_ns, stat.st_size, schemas)
        return schemas

    @staticmethod
    def _load(path: Path, server: str, size: int) -> ToolSchemas:
        if size > MAX_PIN_BYTES:
            logger.error("pin file for MCP server %s exceeds %d bytes", server, MAX_PIN_BYTES)
            raise PinFileError
        try:
            pinned = ListToolsResult.model_validate_json(path.read_bytes())
        except (OSError, ValidationError):
            logger.exception("invalid pin file for MCP server %s", server)
            raise PinFileError from None
        return MappingProxyType({tool.name: tool.input_schema for tool in pinned.tools})
