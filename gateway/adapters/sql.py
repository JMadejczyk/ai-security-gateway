"""Tables a SQL statement reads, for the sql MCP adapter's resources.

SEAM: this is the minimal subset the adapter needs to name resources, not the ``sql_guard``
control. ``sql_guard`` (mandatory, stage 5-10) replaces it with the full allowlist: supported
functions, ``EXPLAIN`` cost, forced ``LIMIT``, timeouts and the rewritten query that executes.
Until then the rule is: exactly one plain ``SELECT`` (Postgres dialect), no data-modifying or
locking clause anywhere in it (CTEs included), every referenced table resolvable to
``schema.table``, and none of the session-settings functions that could change the row-level
security principal mid-statement.
"""

from typing import Final

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers

from gateway.errors import RejectionError

DEFAULT_SCHEMA: Final = "public"
_DIALECT: Final = "postgres"
_FORBIDDEN_NODES: Final = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Command,
    exp.Copy,
    exp.Into,  # SELECT ... INTO creates a table
    exp.Lock,  # FOR UPDATE / FOR SHARE
)
# The RLS principal is a transaction-local setting; a statement must never read or change it.
_FORBIDDEN_FUNCTIONS: Final = frozenset({"set_config", "current_setting"})


class UnsupportedSqlError(RejectionError):
    def __init__(self) -> None:
        super().__init__("unsupported_sql", "only a single plain SELECT is supported")


def _function_name(node: exp.Func) -> str:
    return (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()


def _parse_select(sql: str) -> exp.Select:
    try:
        statements = [s for s in sqlglot.parse(sql, read=_DIALECT) if s is not None]
    except SqlglotError:
        raise UnsupportedSqlError from None
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        raise UnsupportedSqlError
    statement = normalize_identifiers(statements[0], dialect=_DIALECT)
    if statement.find(*_FORBIDDEN_NODES) is not None:
        raise UnsupportedSqlError
    if any(_function_name(f) in _FORBIDDEN_FUNCTIONS for f in statement.find_all(exp.Func)):
        raise UnsupportedSqlError
    return statement


def referenced_tables(sql: str) -> tuple[str, ...]:
    """``schema.table`` for every table the statement reads, sorted and unique.

    Unqualified names resolve to ``public`` (the gateway does not know the upstream's
    ``search_path``; a grant on ``sales.*`` therefore never covers an unqualified name).
    Raises `UnsupportedSqlError` for anything outside the subset described above.
    """
    statement = _parse_select(sql)
    cte_names = {cte.alias_or_name for cte in statement.find_all(exp.CTE)}
    tables: set[str] = set()
    for table in statement.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier) or not table.name or table.catalog:
            raise UnsupportedSqlError  # table functions, catalog-qualified names
        if not table.db and table.name in cte_names:
            continue  # a reference to a CTE defined in this statement, not a table
        tables.add(f"{table.db or DEFAULT_SCHEMA}.{table.name}")
    if not tables:
        raise UnsupportedSqlError
    return tuple(sorted(tables))
