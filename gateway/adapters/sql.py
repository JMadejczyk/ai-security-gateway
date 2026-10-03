"""Tables a SQL statement reads, for the sql MCP adapter's resources.

SEAM: this is the minimal subset the adapter needs to name resources, not the ``sql_guard``
control. ``sql_guard`` (mandatory, stage 5-10) replaces it with the full allowlist, properly
scoped name resolution, ``EXPLAIN`` cost, forced ``LIMIT``, timeouts and the rewritten query
that executes. Until then everything outside a small, fail-closed subset is refused:

- exactly one plain ``SELECT`` (Postgres dialect); no data-modifying or locking clause, no
  ``SELECT INTO``, and no CTEs at all (a CTE name can shadow a real table in another scope);
- only allowlisted functions (aggregates and a few scalar, string and date functions). Any
  other call is refused, including unknown ones and anything that takes SQL text
  (``query_to_xml``, ``dblink``...), which would read tables this check never sees;
- every table schema-qualified (the upstream's ``search_path`` is unknown here, and the SQL is
  forwarded unchanged), outside ``pg_catalog``, ``information_schema`` and ``pg_*`` schemas,
  with plain identifiers only (``[a-z_][a-z0-9_]*`` after Postgres case folding), so that two
  different tables can never map to the same resource.
"""

import re
from enum import StrEnum
from typing import Final

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers

from gateway.errors import RejectionError

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
    exp.With,  # no CTEs until sql_guard resolves names per scope
    exp.CTE,
)
# Reviewed allowlist of sqlglot function nodes (Postgres spelling in the comment). Anything
# else, `exp.Anonymous` (an unknown function) included, is refused.
_ALLOWED_FUNCTIONS: Final[tuple[type[exp.Func], ...]] = (
    exp.Count,  # count
    exp.Sum,  # sum
    exp.Avg,  # avg
    exp.Min,  # min
    exp.Max,  # max
    exp.Lower,  # lower
    exp.Upper,  # upper
    exp.Length,  # length, char_length
    exp.Coalesce,  # coalesce
    exp.Nullif,  # nullif
    exp.Round,  # round
    exp.Abs,  # abs
    exp.TimestampTrunc,  # date_trunc
    exp.Extract,  # extract(field FROM ...)
    exp.CurrentTimestamp,  # now(), current_timestamp
    exp.CurrentDate,  # current_date
    exp.Cast,  # cast(... AS ...), ::
    exp.Case,  # CASE WHEN ... (an expression, sqlglot models it as a function)
    exp.If,  # the WHEN branches of CASE
)
_IDENTIFIER: Final = re.compile(r"[a-z_][a-z0-9_]*")
_SYSTEM_SCHEMAS: Final = frozenset({"pg_catalog", "information_schema"})


class SqlRefusal(StrEnum):
    """Why a statement is outside the subset (the message; the reason code is one)."""

    NOT_A_PLAIN_SELECT = "only a single plain SELECT is supported"
    FUNCTION = "function not supported"
    UNQUALIFIED_TABLE = "tables must be schema-qualified"
    IDENTIFIER = "table names must be plain lower-case identifiers"
    SYSTEM_CATALOG = "system catalogs are not supported"


class UnsupportedSqlError(RejectionError):
    def __init__(self, refusal: SqlRefusal = SqlRefusal.NOT_A_PLAIN_SELECT) -> None:
        super().__init__("unsupported_sql", refusal.value)


def _parse_select(sql: str) -> exp.Select:
    try:
        statements = [s for s in sqlglot.parse(sql, read=_DIALECT) if s is not None]
    except SqlglotError:
        raise UnsupportedSqlError from None
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        raise UnsupportedSqlError
    statement = normalize_identifiers(statements[0], dialect=_DIALECT)  # Postgres case folding
    if statement.find(*_FORBIDDEN_NODES) is not None:
        raise UnsupportedSqlError
    if any(not isinstance(f, _ALLOWED_FUNCTIONS) for f in statement.find_all(exp.Func)):
        raise UnsupportedSqlError(SqlRefusal.FUNCTION)
    return statement


def _resource_name(table: exp.Table) -> str:
    """``schema.table`` for one physical table reference, or `UnsupportedSqlError`."""
    if not isinstance(table.this, exp.Identifier) or table.catalog:
        raise UnsupportedSqlError  # table functions, catalog-qualified names
    if not table.db:
        raise UnsupportedSqlError(SqlRefusal.UNQUALIFIED_TABLE)
    schema, name = table.db, table.name
    if not (_IDENTIFIER.fullmatch(schema) and _IDENTIFIER.fullmatch(name)):
        raise UnsupportedSqlError(SqlRefusal.IDENTIFIER)
    if schema in _SYSTEM_SCHEMAS or schema.startswith("pg_"):
        raise UnsupportedSqlError(SqlRefusal.SYSTEM_CATALOG)
    return f"{schema}.{name}"


def referenced_tables(sql: str) -> tuple[str, ...]:
    """``schema.table`` for every table the statement reads, sorted and unique.

    Raises `UnsupportedSqlError` for anything outside the subset described above.
    """
    statement = _parse_select(sql)
    tables = {_resource_name(table) for table in statement.find_all(exp.Table)}
    if not tables:
        raise UnsupportedSqlError
    return tuple(sorted(tables))
