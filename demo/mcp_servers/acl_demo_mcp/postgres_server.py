"""`mcp-postgres`: trusted SQL upstream that owns the database transaction (SPEC: row-level
filtering).

Per call: verify the gateway's `X-ACL-Principal` assertion (fail closed), take one pooled
connection, begin a read-only transaction, run the transaction-local
`set_config('app.user_id', $1, true)`, execute the single statement the gateway's `sql_guard`
approved, and commit or roll back before the connection goes back to the pool.

The statement is sent with binary results, which forces the extended query protocol: Postgres
then rejects anything but exactly one statement ("cannot insert multiple commands into a prepared
statement"), so a smuggled `; SELECT set_config(...)` cannot run after the approved SELECT. A raw
cursor ($1-style placeholders) leaves `%` in the agent's SQL untouched. The connection role is a
non-owner without BYPASSRLS, the tables FORCE RLS and the transaction is read-only
(see demo/db/sql/01_schema.sql).

Not covered here: a single SELECT that itself calls set_config()/current_setting(). Rejecting
those is the gateway's mandatory sql_guard (SPEC: sql_guard); this server trusts its approval.
"""

from __future__ import annotations

from typing import Any, Final, LiteralString, cast

import anyio
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from psycopg import AsyncConnection, AsyncRawCursor
from psycopg.rows import DictRow, TupleRow, dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from acl_demo_mcp.principal import PrincipalError, PrincipalVerifier

_SET_PRINCIPAL: Final = "SELECT set_config('app.user_id', %s, true)"
_SET_TIMEOUT: Final = "SELECT set_config('statement_timeout', %s, true)"


class PrincipalRejectedError(ToolError):
    def __init__(self) -> None:
        super().__init__("request rejected: no valid principal")


class PostgresSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    host: str = "postgres"
    port: int = Field(default=5432, gt=0, lt=65536)
    dbname: str = "acl_demo"
    user: str = "acl_app"
    password: SecretStr
    statement_timeout_ms: int = Field(default=3000, gt=0)
    max_rows: int = Field(default=1000, gt=0)
    pool_max_size: int = Field(default=4, gt=0)


async def _configure_connection(conn: AsyncConnection[Any]) -> None:
    await conn.set_read_only(True)


class QueryRunner:
    """Runs one approved statement per call inside a principal-scoped transaction."""

    def __init__(self, settings: PostgresSettings, verifier: PrincipalVerifier) -> None:
        self._settings = settings
        self._verifier = verifier
        self._pool: AsyncConnectionPool[AsyncConnection[TupleRow]] = AsyncConnectionPool(
            kwargs={
                "host": settings.host,
                "port": settings.port,
                "dbname": settings.dbname,
                "user": settings.user,
                "password": settings.password.get_secret_value(),
                "application_name": "mcp-postgres",
            },
            min_size=1,
            max_size=settings.pool_max_size,
            configure=_configure_connection,
            open=False,
        )
        self._opened = False
        self._open_lock = anyio.Lock()

    async def _ensure_open(self) -> AsyncConnectionPool[AsyncConnection[TupleRow]]:
        async with self._open_lock:
            if not self._opened:
                await self._pool.open()
                self._opened = True
        return self._pool

    async def query(self, sql: str, ctx: Context) -> list[dict[str, Any]]:
        """Run one read-only SQL SELECT against the sales database; rows are filtered per user."""
        try:
            principal = self._verifier.verify(ctx.headers)
        except PrincipalError as exc:
            raise PrincipalRejectedError from exc
        pool = await self._ensure_open()
        async with pool.connection() as conn, conn.transaction():
            await conn.execute(_SET_PRINCIPAL, (principal,))
            await conn.execute(_SET_TIMEOUT, (str(self._settings.statement_timeout_ms),))
            async with AsyncRawCursor(conn, row_factory=dict_row) as cur:
                # The gateway's sql_guard approved this exact text; it is never interpolated.
                # binary=True forces the extended protocol, i.e. exactly one statement.
                await cur.execute(cast(LiteralString, sql), binary=True)
                if cur.description is None:
                    return []
                rows: list[DictRow] = await cur.fetchmany(self._settings.max_rows)
        return [dict(row) for row in rows]


def build_postgres_server(settings: PostgresSettings, verifier: PrincipalVerifier) -> MCPServer:
    runner = QueryRunner(settings, verifier)
    server = MCPServer(name="mcp-postgres")
    server.tool(
        name="query",
        annotations=ToolAnnotations(
            title="SQL query",
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )(runner.query)
    return server
