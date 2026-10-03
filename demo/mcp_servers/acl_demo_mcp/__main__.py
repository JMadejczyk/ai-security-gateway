"""Entry point: `python -m acl_demo_mcp {postgres|fetch|files}` serves streamable HTTP."""

from __future__ import annotations

import os
import sys
from enum import StrEnum
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from pydantic import SecretStr

from acl_demo_mcp.config import ListenSettings, required_env
from acl_demo_mcp.fetch_server import build_fetch_server
from acl_demo_mcp.files_server import FilesSettings, build_files_server
from acl_demo_mcp.postgres_server import PostgresSettings, build_postgres_server
from acl_demo_mcp.principal import PrincipalVerifier


class ServerKind(StrEnum):
    POSTGRES = "postgres"
    FETCH = "fetch"
    FILES = "files"


def _postgres() -> MCPServer:
    settings = PostgresSettings(
        host=os.environ.get("PGHOST", "postgres"),
        port=int(os.environ.get("PGPORT", "5432")),
        dbname=os.environ.get("PGDATABASE", "acl_demo"),
        user=os.environ.get("PGUSER", "acl_app"),
        password=SecretStr(required_env("ACL_APP_DB_PASSWORD")),
    )
    verifier = PrincipalVerifier(
        key=required_env("ACL_INTERNAL_KEY").encode(),
        audience=os.environ.get("ACL_PRINCIPAL_AUDIENCE", "mcp-postgres"),
        issuer=os.environ.get("ACL_PRINCIPAL_ISSUER", "ai-control-layer"),
    )
    return build_postgres_server(settings, verifier)


def _files() -> MCPServer:
    return build_files_server(
        FilesSettings(root=Path(os.environ.get("REPORTS_ROOT", "/data/reports")))
    )


def build(kind: ServerKind) -> MCPServer:
    match kind:
        case ServerKind.POSTGRES:
            return _postgres()
        case ServerKind.FETCH:
            return build_fetch_server()
        case ServerKind.FILES:
            return _files()


def main(argv: list[str]) -> None:
    if len(argv) != 1 or argv[0] not in set(ServerKind):
        choices = ", ".join(kind.value for kind in ServerKind)
        sys.exit(f"usage: python -m acl_demo_mcp {{{choices}}}")
    listen = ListenSettings.from_env()
    build(ServerKind(argv[0])).run(
        "streamable-http",
        host=listen.host,
        port=listen.port,
        streamable_http_path=listen.path,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
