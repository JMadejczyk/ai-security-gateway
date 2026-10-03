"""SQL analysis shared by the sql MCP adapter (resources) and the ``sql_guard`` control (rewrite).

`QueryPlan.parse` parses a statement once and refuses everything outside a small, fail-closed
subset (SPEC "Control catalog" → ``sql_guard``):

- exactly one plain ``SELECT`` (Postgres dialect); no data-modifying or locking clause, no
  ``SELECT INTO``, no set operations and no CTEs at all (a CTE name can shadow a real table in
  another scope, so refusing them is stricter than refusing only data-modifying CTEs);
- only allowlisted functions (aggregates and a few scalar, string and date functions). Any
  other call is refused, including unknown ones, anything that takes SQL text
  (``query_to_xml``, ``dblink``...), which would read tables this check never sees, and every
  session or settings function (``set_config``, ``current_setting``, ``acl.set_principal``);
- every table schema-qualified (the upstream's ``search_path`` is unknown here), outside
  ``pg_catalog``, ``information_schema`` and ``pg_*`` schemas, with plain identifiers only
  (``[a-z_][a-z0-9_]*`` after Postgres case folding), so that two different tables can never
  map to the same resource;
- the outermost row cap (``LIMIT`` or ``FETCH FIRST``), when present, is an integer literal
  (or ``LIMIT ALL`` / ``LIMIT NULL``, which mean no cap).

The plan's `QueryPlan.sql` is sqlglot's Postgres rendering of the parsed tree with comments
dropped, and that rendering is what ``sql_guard`` sends upstream. Executing the analyzer's own
rendering, rather than the agent's text, closes the class of parser differentials (comments,
quoting, escapes) where Postgres would read something this module did not see.
`QueryPlan.with_row_limit` caps the outermost ``SELECT`` and re-parses the result, refusing it
unless it reads exactly the same tables.
"""

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Self

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
    exp.With,  # no CTEs: names would need resolving per scope
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
    ROW_LIMIT = "LIMIT and FETCH FIRST must be integer literals"
    REWRITE = "the rewritten statement does not read the same tables"


class UnsupportedSqlError(RejectionError):
    def __init__(self, refusal: SqlRefusal = SqlRefusal.NOT_A_PLAIN_SELECT) -> None:
        super().__init__("unsupported_sql", refusal.value)


@dataclass(frozen=True, slots=True)
class QueryPlan:
    """One statement inside the supported subset: the tables it reads and its canonical text.

    Build it with `parse`; the constructor trusts its arguments.
    """

    tables: tuple[str, ...]  # ``schema.table``, sorted and unique, never empty
    row_limit: int | None  # the outermost cap; None = unbounded
    statement: exp.Select  # normalized (Postgres case folding); never mutated

    @classmethod
    def parse(cls, sql: str) -> Self:
        """Analyze ``sql``; raises `UnsupportedSqlError` for anything outside the subset."""
        statement = _parse_select(sql)
        tables = {_resource_name(table) for table in statement.find_all(exp.Table)}
        if not tables:
            raise UnsupportedSqlError
        return cls(
            tables=tuple(sorted(tables)), row_limit=_row_limit(statement), statement=statement
        )

    @property
    def sql(self) -> str:
        """The statement as Postgres will run it: sqlglot's rendering, comments dropped."""
        return self.statement.sql(dialect=_DIALECT, comments=False)

    def with_row_limit(self, limit: int) -> Self:
        """This plan with the outermost ``SELECT`` capped at ``limit`` rows.

        A cap already at or below ``limit`` is kept (as is any ``OFFSET``); anything else,
        including no cap, becomes ``LIMIT <limit>``. Subqueries are never touched. The result
        is re-parsed from its own text and refused unless it reads exactly the same tables.
        """
        if limit <= 0:
            msg = f"row limit must be positive, got {limit}"
            raise ValueError(msg)
        if self.row_limit is not None and self.row_limit <= limit:
            return self
        capped = self.statement.copy()
        capped.set("limit", exp.Limit(expression=exp.Literal.number(limit)))
        rewritten = type(self).parse(capped.sql(dialect=_DIALECT, comments=False))
        if rewritten.tables != self.tables or rewritten.row_limit != limit:
            raise UnsupportedSqlError(SqlRefusal.REWRITE)
        return rewritten


def referenced_tables(sql: str) -> tuple[str, ...]:
    """``schema.table`` for every table the statement reads, sorted and unique.

    Raises `UnsupportedSqlError` for anything outside the subset described above.
    """
    return QueryPlan.parse(sql).tables


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


def _integer(node: object) -> int:
    """The value of a non-negative integer literal, or `UnsupportedSqlError`."""
    if isinstance(node, exp.Literal) and not node.is_string and node.name.isdigit():
        return int(node.name)
    raise UnsupportedSqlError(SqlRefusal.ROW_LIMIT)


def _row_limit(statement: exp.Select) -> int | None:
    """The outermost ``SELECT``'s row cap. ``LIMIT ALL`` parses as no limit at all."""
    match statement.args.get("limit"):
        case None:
            return None
        case exp.Limit() as limit if isinstance(limit.expression, exp.Null):
            return None
        case exp.Limit() as limit:
            return _integer(limit.expression)
        case exp.Fetch() as fetch:
            count = fetch.args.get("count")
            value = 1 if count is None else _integer(count)  # FETCH FIRST ROW ONLY
            options = fetch.args.get("limit_options")
            if isinstance(options, exp.LimitOptions) and (
                options.args.get("with_ties") or options.args.get("percent")
            ):
                return None  # WITH TIES or PERCENT can return more than `count` rows
            return value
        case _:
            raise UnsupportedSqlError(SqlRefusal.ROW_LIMIT)
