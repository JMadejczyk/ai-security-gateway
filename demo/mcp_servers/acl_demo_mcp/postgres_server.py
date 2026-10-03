"""`mcp-postgres`: trusted SQL upstream that owns the database transaction (SPEC: row-level
filtering).

Every call verifies the gateway's `X-ACL-Principal` assertion and its signed execution limits
(fail closed without either), checks the statement is one plain `SELECT` (`sql_check`), takes
one pooled connection and runs inside one read-only transaction:

1. `SELECT acl.set_principal($1)`: the transaction-local `app.user_id` that RLS filters on.
   `set_config` itself is not executable by this role (demo/db/sql/01_schema.sql), and
   `acl.set_principal` refuses a second call in the same transaction, so even a statement
   that slipped past the gateway's `sql_guard` cannot change the principal;
2. `SET LOCAL statement_timeout` and `SET LOCAL lock_timeout` from the limits claim;
3. the statement, then commit or roll back before the connection goes back to the pool.

Tools:

- `query`: runs the exact statement `sql_guard` approved through a server-side cursor
  (`DECLARE ... CURSOR FOR`, always sent with the extended protocol, so exactly one statement,
  and necessarily a query), fetches at most `max_rows + 1` rows and refuses the call when there
  are more, or when the serialized result exceeds `max_result_bytes`;
- `explain`: the gateway's planner, absent from the gateway's operator mapping so agents never
  reach it. Same principal and transaction setup; runs `EXPLAIN (FORMAT JSON) <statement>`
  (never `ANALYZE`, the statement is not executed) and returns the plan's top-level
  `Total Cost`.

Postgres errors become a generic tool error carrying only the SQLSTATE: their text can quote
the statement. The connection role is a non-owner without BYPASSRLS and every table FORCEs RLS.
"""

from __future__ import annotations

import secrets
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Final, LiteralString, cast

import anyio
import psycopg
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from psycopg import AsyncConnection, AsyncRawCursor, AsyncRawServerCursor, errors, sql
from psycopg.rows import DictRow, TupleRow, dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, ConfigDict, Field, SecretStr, TypeAdapter, ValidationError

from acl_demo_mcp.principal import ExecutionLimits, PrincipalError, PrincipalVerifier
from acl_demo_mcp.sql_check import NotASelectError, require_single_select

_SET_PRINCIPAL: Final = "SELECT acl.set_principal(%s)"
# SET LOCAL takes no bind parameters; the values are validated integers from the signed claim.
_SET_STATEMENT_TIMEOUT: Final = sql.SQL("SET LOCAL statement_timeout = {}")
_SET_LOCK_TIMEOUT: Final = sql.SQL("SET LOCAL lock_timeout = {}")
_EXPLAIN: Final = "EXPLAIN (FORMAT JSON) "
_ROWS: Final = TypeAdapter(list[dict[str, Any]])


class PrincipalRejectedError(ToolError):
    def __init__(self) -> None:
        super().__init__("request rejected: no valid principal")


class StatementRejectedError(ToolError):
    def __init__(self) -> None:
        super().__init__("request rejected: only a single read-only SELECT is accepted")


class ResultLimitError(ToolError):
    def __init__(self, what: str, limit: int) -> None:
        super().__init__(f"result refused: more than {limit} {what}")


class StatementTimeoutError(ToolError):
    def __init__(self) -> None:
        super().__init__("statement canceled: execution limit exceeded")


class StatementFailedError(ToolError):
    def __init__(self, sqlstate: str | None) -> None:
        super().__init__(f"statement failed (SQLSTATE {sqlstate or 'unknown'})")


class PostgresSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    host: str = "postgres"
    port: int = Field(default=5432, gt=0, lt=65536)
    dbname: str = "acl_demo"
    user: str = "acl_app"
    password: SecretStr
    pool_max_size: int = Field(default=4, gt=0)


class PlanCost(BaseModel):
    """What `explain` returns: the planner's estimated cost of the whole statement."""

    total_cost: float = Field(ge=0.0)


class _PlanNode(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    total_cost: float = Field(alias="Total Cost", ge=0.0, allow_inf_nan=False)


class _ExplainEntry(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    plan: _PlanNode = Field(alias="Plan")


_EXPLAIN_OUTPUT: Final = TypeAdapter(list[_ExplainEntry])


async def _configure_connection(conn: AsyncConnection[Any]) -> None:
    await conn.set_read_only(True)


class QueryRunner:
    """Runs one statement per call inside a principal-scoped, limit-bounded transaction."""

    def __init__(self, settings: PostgresSettings, verifier: PrincipalVerifier) -> None:
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

    async def aclose(self) -> None:
        async with self._open_lock:
            if self._opened:
                await self._pool.close()
                self._opened = False

    async def _ensure_open(self) -> AsyncConnectionPool[AsyncConnection[TupleRow]]:
        async with self._open_lock:
            if not self._opened:
                await self._pool.open()
                self._opened = True
        return self._pool

    @asynccontextmanager
    async def _transaction(
        self, statement: str, ctx: Context
    ) -> AsyncGenerator[tuple[AsyncConnection[TupleRow], ExecutionLimits]]:
        """Verified principal and limits, a checked statement, one set-up transaction."""
        try:
            claims = self._verifier.verify(ctx.headers)
            limits = claims.require_limits()
        except PrincipalError as exc:
            raise PrincipalRejectedError from exc
        try:
            require_single_select(statement)
        except NotASelectError as exc:
            raise StatementRejectedError from exc
        pool = await self._ensure_open()
        try:
            async with pool.connection() as conn, conn.transaction():
                await conn.execute(_SET_PRINCIPAL, (claims.sub,))
                timeout = sql.Literal(limits.stmt_timeout_ms)
                await conn.execute(_SET_STATEMENT_TIMEOUT.format(timeout))
                await conn.execute(_SET_LOCK_TIMEOUT.format(timeout))
                yield conn, limits
        except errors.QueryCanceled as exc:
            raise StatementTimeoutError from exc
        except psycopg.Error as exc:
            raise StatementFailedError(exc.sqlstate) from exc

    async def query(self, sql: str, ctx: Context) -> list[dict[str, Any]]:
        """Run one read-only SQL SELECT against the sales database; rows are filtered per user."""
        name = f"acl_{secrets.token_hex(8)}"
        async with (
            self._transaction(sql, ctx) as (conn, limits),
            AsyncRawServerCursor(conn, name, row_factory=dict_row) as cur,
        ):
            # The gateway's sql_guard approved this exact text; it is never interpolated.
            await cur.execute(cast(LiteralString, sql))
            rows: list[DictRow] = await cur.fetchmany(limits.max_rows + 1)
        if len(rows) > limits.max_rows:
            raise ResultLimitError("rows", limits.max_rows)
        result = [dict(row) for row in rows]
        if len(_ROWS.dump_json(result)) > limits.max_result_bytes:
            raise ResultLimitError("bytes", limits.max_result_bytes)
        return result

    async def explain(self, sql: str, ctx: Context) -> PlanCost:
        """Planner cost of one SELECT, without running it (gateway use only)."""
        async with (
            self._transaction(sql, ctx) as (conn, _limits),
            AsyncRawCursor(conn) as cur,
        ):
            # binary=True forces the extended protocol: exactly one statement.
            await cur.execute(cast(LiteralString, _EXPLAIN + sql), binary=True)
            row = await cur.fetchone()
        try:
            [entry] = _EXPLAIN_OUTPUT.validate_python(row[0] if row else None)
        except (ValidationError, ValueError) as exc:
            raise StatementFailedError(None) from exc
        return PlanCost(total_cost=entry.plan.total_cost)


def build_postgres_server(settings: PostgresSettings, verifier: PrincipalVerifier) -> MCPServer:
    runner = QueryRunner(settings, verifier)
    server = MCPServer(name="mcp-postgres")
    read_only = ToolAnnotations(
        read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
    )
    server.tool(name="query", annotations=read_only.model_copy(update={"title": "SQL query"}))(
        runner.query
    )
    server.tool(
        name="explain", annotations=read_only.model_copy(update={"title": "SQL plan cost"})
    )(runner.explain)
    return server
