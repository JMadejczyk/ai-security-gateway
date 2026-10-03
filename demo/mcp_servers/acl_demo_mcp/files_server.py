"""`mcp-files`: writes reports under one root directory (a volume mounted at /data/reports).

Report names are flat file names: no directories, no absolute paths, no `..`. The file is created
with O_CREAT | O_EXCL, which never follows a symlink and never overwrites: a symlink planted in
the root cannot redirect the write, and an existing report is never replaced (the tool is
additive only, matching its `destructiveHint: false` annotation).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", re.ASCII)
_FILE_MODE = 0o640


class ReportPathError(ValueError):
    """A report cannot be written at the requested name."""


class InvalidReportNameError(ReportPathError):
    def __init__(self) -> None:
        super().__init__("report name must be a plain file name: letters, digits, '.', '_', '-'")


class ReportEscapesRootError(ReportPathError):
    def __init__(self) -> None:
        super().__init__("report name escapes the report root")


class ReportTooLargeError(ReportPathError):
    def __init__(self, limit: int) -> None:
        super().__init__(f"report exceeds {limit} bytes")


class ReportExistsError(ReportPathError):
    def __init__(self, name: str) -> None:
        super().__init__(f"report {name!r} already exists")


class FilesSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    root: Path = Path("/data/reports")
    max_bytes: int = Field(default=1024 * 1024, gt=0)


class ReportStore:
    """Path-root-enforcing writer for report files."""

    def __init__(self, settings: FilesSettings) -> None:
        self._root = settings.root.resolve(strict=True)
        self._max_bytes = settings.max_bytes

    def resolve(self, name: str) -> Path:
        """Map a report name to a path directly inside the root, or raise `ReportPathError`."""
        if not _NAME_RE.fullmatch(name):
            raise InvalidReportNameError
        path = self._root / name
        if path.parent != self._root:
            raise ReportEscapesRootError
        return path

    def write(self, name: str, content: str) -> int:
        data = content.encode("utf-8")
        if len(data) > self._max_bytes:
            raise ReportTooLargeError(self._max_bytes)
        path = self.resolve(name)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        try:
            fd = os.open(path, flags, _FILE_MODE)
        except FileExistsError as exc:  # also raised when the name is an existing symlink
            raise ReportExistsError(name) from exc
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        return len(data)


class ReportTools:
    def __init__(self, store: ReportStore) -> None:
        self._store = store

    def write_report(self, name: str, content: str) -> str:
        """Create a new text report `name` (a plain file name); never replaces an existing one."""
        try:
            written = self._store.write(name, content)
        except ReportPathError as exc:
            raise ToolError(str(exc)) from exc
        return f"wrote {written} bytes to reports/{name}"


def build_files_server(settings: FilesSettings | None = None) -> MCPServer:
    tools = ReportTools(ReportStore(settings or FilesSettings()))
    server = MCPServer(name="mcp-files")
    server.tool(
        name="write_report",
        annotations=ToolAnnotations(
            title="Write report",
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=False,
        ),
    )(tools.write_report)
    return server
