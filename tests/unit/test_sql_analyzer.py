"""`QueryPlan`: the supported SELECT subset, the forced LIMIT and the canonical text that runs."""

import pytest

from gateway.adapters.sql import QueryPlan, SqlRefusal, UnsupportedSqlError

FORCE = 500

# Every statement the adapter refused before sql_guard existed, plus the cases sql_guard adds.
REJECTED = [
    "DELETE FROM sales.orders",
    "DROP TABLE sales.orders",
    "UPDATE sales.orders SET total = 0",
    "INSERT INTO sales.orders VALUES (1)",
    "SELECT 1 FROM sales.orders; SELECT 2 FROM sales.customers",
    "WITH d AS (DELETE FROM sales.orders RETURNING *) SELECT * FROM d",
    "SELECT * INTO stolen FROM sales.customers",
    "SELECT * FROM sales.orders FOR UPDATE",
    "SELECT * FROM sales.orders FOR SHARE",
    "SELECT set_config('app.user_id', 'root@demo', true) FROM sales.customers",
    "SELECT current_setting('app.user_id')",
    "SELECT current_setting('app.user_id') FROM sales.customers",
    "SELECT acl.set_principal('root@demo') FROM sales.customers",
    "SELECT * FROM sales.customers WHERE id IN (SELECT acl.set_principal('root@demo'))",
    "SELECT pg_catalog.set_config('statement_timeout', '0', true) FROM sales.orders",
    "SELECT 1",
    "SELECT * FROM generate_series(1, 3)",
    "SELECT * FROM otherdb.sales.orders",
    "COPY sales.orders TO STDOUT",
    "SELECT 1 FROM sales.orders UNION SELECT 2 FROM sales.payments",
    "this is not sql (((",
    "",
    "SELECT query_to_xml($$SELECT * FROM sales.payments$$, true, false, '') FROM sales.customers",
    "SELECT pg_read_file('/etc/passwd') FROM sales.orders",
    "SELECT dblink('host=x', 'SELECT 1') FROM sales.orders",
    "SELECT made_up_function(total) FROM sales.orders",
    "SELECT pg_sleep(10) FROM sales.orders",
    "WITH t AS (SELECT * FROM sales.orders) SELECT * FROM t",
    "WITH unused AS (SELECT 1) SELECT * FROM sales.customers",
    "SELECT * FROM customers",
    "SELECT * FROM pg_catalog.pg_class",
    "SELECT * FROM information_schema.tables",
    "SELECT * FROM pg_toast.pg_toast_1",
    'SELECT * FROM "public.sales".customers',
    'SELECT * FROM public."sales.customers"',
    'SELECT * FROM sales."Customers"',
    'SELECT * FROM sales."cust omers"',
    # Row caps that are not integer literals.
    "SELECT * FROM sales.orders LIMIT (SELECT COUNT(*) FROM sales.payments)",
    "SELECT * FROM sales.orders LIMIT 1 + 1",
    "SELECT * FROM sales.orders LIMIT $1",
    "SELECT * FROM sales.orders LIMIT '5'",
    "SELECT * FROM sales.orders LIMIT -1",
    "SELECT * FROM sales.orders LIMIT 1.5",
    "SELECT * FROM sales.orders FETCH FIRST (SELECT 1) ROWS ONLY",
]


@pytest.mark.parametrize("sql", REJECTED)
def test_statements_outside_the_subset_are_refused(sql):
    with pytest.raises(UnsupportedSqlError) as exc:
        QueryPlan.parse(sql)
    assert exc.value.reason_code == "unsupported_sql"


def test_session_functions_say_why():
    with pytest.raises(UnsupportedSqlError) as exc:
        QueryPlan.parse("SELECT set_config('app.user_id', 'x', true) FROM sales.customers")
    assert exc.value.message == SqlRefusal.FUNCTION


def test_non_literal_limits_say_why():
    with pytest.raises(UnsupportedSqlError) as exc:
        QueryPlan.parse("SELECT * FROM sales.orders LIMIT 1 + 1")
    assert exc.value.message == SqlRefusal.ROW_LIMIT


@pytest.mark.parametrize(
    ("sql", "rewritten"),
    [
        (
            "SELECT COUNT(*) FROM sales.customers",
            "SELECT COUNT(*) FROM sales.customers LIMIT 500",
        ),
        ("SELECT * FROM sales.orders LIMIT 10", "SELECT * FROM sales.orders LIMIT 10"),
        ("SELECT * FROM sales.orders LIMIT 500", "SELECT * FROM sales.orders LIMIT 500"),
        ("SELECT * FROM sales.orders LIMIT 100000", "SELECT * FROM sales.orders LIMIT 500"),
        ("SELECT * FROM sales.orders LIMIT 501", "SELECT * FROM sales.orders LIMIT 500"),
        (
            "SELECT * FROM sales.orders LIMIT 100000 OFFSET 20",
            "SELECT * FROM sales.orders LIMIT 500 OFFSET 20",
        ),
        (
            "SELECT * FROM sales.orders LIMIT 5 OFFSET 20",
            "SELECT * FROM sales.orders LIMIT 5 OFFSET 20",
        ),
        ("SELECT * FROM sales.orders OFFSET 20", "SELECT * FROM sales.orders LIMIT 500 OFFSET 20"),
        ("SELECT * FROM sales.orders LIMIT ALL", "SELECT * FROM sales.orders LIMIT 500"),
        ("SELECT * FROM sales.orders LIMIT NULL", "SELECT * FROM sales.orders LIMIT 500"),
        (
            "SELECT * FROM sales.orders FETCH FIRST 3 ROWS ONLY",
            "SELECT * FROM sales.orders FETCH FIRST 3 ROWS ONLY",
        ),
        (
            "SELECT * FROM sales.orders FETCH FIRST 9000 ROWS ONLY",
            "SELECT * FROM sales.orders LIMIT 500",
        ),
        (  # WITH TIES can return more rows than its count: capped
            "SELECT * FROM sales.orders ORDER BY amount FETCH FIRST 3 ROWS WITH TIES",
            "SELECT * FROM sales.orders ORDER BY amount LIMIT 500",
        ),
        (  # only the outermost SELECT is capped; subquery limits are the agent's own
            "SELECT * FROM (SELECT * FROM sales.payments LIMIT 100000) p",
            "SELECT * FROM (SELECT * FROM sales.payments LIMIT 100000) AS p LIMIT 500",
        ),
        (
            "SELECT c.id FROM sales.customers c WHERE c.id IN "
            "(SELECT o.customer_id FROM sales.orders o LIMIT 9999) LIMIT 7",
            "SELECT c.id FROM sales.customers AS c WHERE c.id IN "
            "(SELECT o.customer_id FROM sales.orders AS o LIMIT 9999) LIMIT 7",
        ),
    ],
    ids=[
        "no-limit-aggregate",
        "small-limit-kept",
        "equal-limit-kept",
        "large-limit-capped",
        "just-over-capped",
        "large-limit-with-offset",
        "small-limit-with-offset",
        "offset-only",
        "limit-all",
        "limit-null",
        "small-fetch-kept",
        "large-fetch-capped",
        "fetch-with-ties",
        "subquery-limit-untouched",
        "in-subquery-limit-untouched",
    ],
)
def test_outermost_select_is_capped(sql, rewritten):
    plan = QueryPlan.parse(sql)
    capped = plan.with_row_limit(FORCE)
    assert capped.sql == rewritten
    assert capped.tables == plan.tables
    assert capped.row_limit is not None
    assert capped.row_limit <= FORCE


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(*) FROM sales.customers",
        "select c.name, o.amount from Sales.Customers c "
        "join sales.orders o on o.customer_id = c.id",
        "SELECT region, COUNT(*) FROM sales.customers GROUP BY region HAVING COUNT(*) > 1",
        "SELECT * FROM (SELECT * FROM sales.payments) p WHERE p.amount > 10",
        "SELECT COUNT(*) FROM sales.customers c CROSS JOIN sales.orders o "
        "CROSS JOIN sales.payments p",
        "SELECT date_trunc('month', ordered_at), SUM(amount)::text FROM sales.orders GROUP BY 1",
    ],
)
def test_rewritten_sql_reparses_to_the_same_tables(sql):
    plan = QueryPlan.parse(sql)
    capped = plan.with_row_limit(FORCE)
    assert QueryPlan.parse(capped.sql).tables == plan.tables
    assert QueryPlan.parse(capped.sql).sql == capped.sql  # the rendering is a fixed point


def test_canonical_text_drops_comments():
    plan = QueryPlan.parse("SELECT id /* hi */ FROM sales.orders -- */ , pg_sleep(1) /*")
    assert plan.sql == "SELECT id FROM sales.orders"
    assert "--" not in plan.with_row_limit(FORCE).sql


def test_quoted_and_folded_names_are_one_table():
    assert QueryPlan.parse('SELECT * FROM "sales"."customers"').tables == ("sales.customers",)
    assert QueryPlan.parse("SELECT * FROM Sales.Customers").tables == ("sales.customers",)


def test_row_limit_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        QueryPlan.parse("SELECT * FROM sales.orders").with_row_limit(0)
