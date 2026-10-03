"""The server's own check that a statement is one plain read-only ``SELECT``.

Defense in depth behind the gateway's mandatory ``sql_guard`` (which owns the full subset:
function allowlist, table resolution, forced ``LIMIT``, plan cost). Here only the shape is
checked, with the same parser and dialect: exactly one statement, a ``SELECT``, and nothing
that writes, locks or creates. The extended query protocol and the read-only transaction
enforce the same again in Postgres.
"""

from __future__ import annotations

from typing import Final

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

_DIALECT: Final = "postgres"
_FORBIDDEN: Final = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Command,
    exp.Copy,
    exp.Into,
    exp.Lock,
)


class NotASelectError(ValueError):
    def __init__(self) -> None:
        super().__init__("only a single read-only SELECT is accepted")


def require_single_select(sql: str) -> None:
    """Raise `NotASelectError` unless ``sql`` is exactly one plain read-only ``SELECT``."""
    try:
        statements = [s for s in sqlglot.parse(sql, read=_DIALECT) if s is not None]
    except SqlglotError:
        raise NotASelectError from None
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        raise NotASelectError
    if statements[0].find(*_FORBIDDEN) is not None:
        raise NotASelectError
