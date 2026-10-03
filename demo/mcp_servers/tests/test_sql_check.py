from __future__ import annotations

import pytest

from acl_demo_mcp.sql_check import NotASelectError, require_single_select


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(*) FROM sales.customers LIMIT 500",
        "SELECT c.name FROM sales.customers AS c JOIN sales.orders AS o ON o.customer_id = c.id",
        "SELECT * FROM (SELECT * FROM sales.payments LIMIT 3) AS p LIMIT 500",
    ],
)
def test_single_selects_are_accepted(sql: str) -> None:
    require_single_select(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "",
        "this is not sql (((",
        "SELECT 1; SELECT 2",
        "SELECT 1 FROM sales.orders; SELECT set_config('app.user_id', 'root@demo', true)",
        "ANALYZE SELECT * FROM sales.orders",
        "ANALYZE sales.orders",
        "EXPLAIN ANALYZE SELECT * FROM sales.orders",
        "(ANALYZE) SELECT * FROM sales.orders",
        "DELETE FROM sales.orders",
        "UPDATE sales.orders SET amount = 0",
        "INSERT INTO sales.orders VALUES (1)",
        "WITH d AS (DELETE FROM sales.orders RETURNING *) SELECT * FROM d",
        "SELECT * INTO stolen FROM sales.customers",
        "SELECT * FROM sales.orders FOR UPDATE",
        "COPY sales.orders TO STDOUT",
        "SET statement_timeout = 0",
        "SELECT 1 UNION SELECT 2",
        "DROP TABLE sales.orders",
    ],
)
def test_everything_else_is_refused(sql: str) -> None:
    with pytest.raises(NotASelectError):
        require_single_select(sql)
