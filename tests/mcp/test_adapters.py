"""MCP adapters: operator mapping → interactions; templates, hosts, paths, SQL tables, schemas."""

from typing import Any

import pytest

from gateway.adapters.mcp import (
    FsMCPAdapter,
    GenericMCPAdapter,
    HttpMCPAdapter,
    SqlMCPAdapter,
    mcp_adapter,
    normalized_path,
    url_host,
)
from gateway.adapters.sql import referenced_tables
from gateway.core.envelope import RawCall
from gateway.core.types import Action, Channel
from gateway.errors import RejectionError
from gateway.policy.permissions import PermissionSet

OPEN_OBJECT: dict[str, Any] = {"type": "object", "additionalProperties": True}
QUERY_SCHEMA = {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}
FETCH_SCHEMA = {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}
WRITE_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "content": {"type": "string"}},
    "required": ["name", "content"],
}


@pytest.fixture
def servers(snapshot):
    return snapshot.policy.upstreams.mcp


@pytest.fixture
def ctx(make_ctx):
    return make_ctx()


def raw(server: str, tool: str, **arguments: Any) -> RawCall:
    return RawCall(channel=Channel.MCP, server=server, data={"name": tool, "arguments": arguments})


def reason(exc: pytest.ExceptionInfo[RejectionError]) -> str:
    return exc.value.reason_code


# ------------------------------------------------------------------------- dispatch


def test_policy_adapter_kind_selects_the_class(servers):
    assert type(mcp_adapter("sales_db", servers["sales_db"], {})) is SqlMCPAdapter
    assert type(mcp_adapter("web", servers["web"], {})) is HttpMCPAdapter
    assert type(mcp_adapter("reports", servers["reports"], {})) is FsMCPAdapter


def test_matches_only_its_own_server(servers):
    adapter = mcp_adapter("web", servers["web"], {"fetch": FETCH_SCHEMA})
    assert adapter.matches(raw("web", "fetch"))
    assert not adapter.matches(raw("reports", "fetch"))
    assert not adapter.matches(RawCall(channel=Channel.LLM, data={}))


# ------------------------------------------------------------------- generic template


@pytest.fixture
def generic(snapshot_from, policy_doc):
    policy_doc["upstreams"]["mcp"]["kb"] = {
        "url": "http://mcp-kb:8000/mcp",
        "adapter": "generic",
        "trust": "internal",
        "tools": {"get_doc": {"action": "read", "resource": "kb:{space}/{id}"}},
    }
    config = snapshot_from(policy_doc).policy.upstreams.mcp["kb"]
    return GenericMCPAdapter("kb", config, {"get_doc": OPEN_OBJECT})


def test_template_fills_each_placeholder(generic, ctx):
    [interaction] = generic.normalize(raw("kb", "get_doc", space="eng", id="42"), ctx)
    assert (interaction.action, interaction.resource) == (Action.READ, "kb:eng/42")
    assert interaction.payload == {"name": "get_doc", "arguments": {"space": "eng", "id": "42"}}
    assert interaction.channel is Channel.MCP


def test_generic_value_is_one_encoded_segment(generic, ctx):
    [interaction] = generic.normalize(raw("kb", "get_doc", space="eng", id="a b/../*"), ctx)
    assert interaction.resource == "kb:eng/a%20b%2F..%2F%2A"
    assert not PermissionSet.parse(["read:kb:eng/a*/x"]).allows(Action.READ, interaction.resource)


@pytest.mark.parametrize(
    "arguments",
    [{"space": "eng"}, {"space": "eng", "id": 42}, {"space": "eng", "id": ""}, {"id": "1"}],
    ids=["missing", "not-a-string", "empty", "other-missing"],
)
def test_missing_or_non_string_argument_is_invalid(generic, ctx, arguments):
    with pytest.raises(RejectionError) as exc:
        generic.normalize(raw("kb", "get_doc", **arguments), ctx)
    assert reason(exc) == "invalid_arguments"


# ------------------------------------------------------------------------ http host


@pytest.mark.parametrize(
    ("url", "host"),
    [
        ("https://Example.COM/a?b=c", "example.com"),
        ("http://example.com:8443/x", "example.com"),
        ("https://faß.de/", "xn--fa-hia.de"),  # IDNA 2008, as httpx sends it (not fass.de)
        ("http://93.184.216.34/", "93.184.216.34"),
        ("http://[2001:db8::1]:80/", "2001:db8::1"),
        ("https://bücher.de/", "xn--bcher-kva.de"),
    ],
)
def test_url_reduced_to_host(url, host):
    assert url_host(url) == host


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/x",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "http://user:pw@example.com/",
        "http://evil.com\\@good.com/",
        "http:///nohost",
        "example.com/path",
        "http://exa mple.com/",
        "http://example.com:99999/",
        "http://ex*ample.com/",
        "http://example.com\n.evil/",
        "http://example.com./",  # a trailing dot would authorize a different spelling
    ],
)
def test_bad_urls_refused(url):
    with pytest.raises(RejectionError) as exc:
        url_host(url)
    assert reason(exc) == "invalid_arguments"


def test_http_adapter_resource_is_the_host(servers, ctx):
    adapter = mcp_adapter("web", servers["web"], {"fetch": FETCH_SCHEMA})
    [interaction] = adapter.normalize(raw("web", "fetch", url="https://news.example.com/x"), ctx)
    assert (interaction.action, interaction.resource) == (Action.READ, "web:news.example.com")


def test_http_adapter_forwards_the_url_it_authorized(servers, ctx):
    """The fetcher gets the canonical URL, so the host it connects to is the one checked."""
    adapter = mcp_adapter("web", servers["web"], {"fetch": FETCH_SCHEMA})
    [interaction] = adapter.normalize(raw("web", "fetch", url="https://Faß.DE/a?b=ç"), ctx)
    assert interaction.resource == "web:xn--fa-hia.de"
    forwarded = interaction.payload["arguments"]["url"]
    assert forwarded.startswith("https://xn--fa-hia.de/a?b=")
    assert url_host(forwarded) == "xn--fa-hia.de"


# --------------------------------------------------------------------------- fs path


@pytest.mark.parametrize(
    "path",
    ["../x", "/etc/passwd", "a\\b", "a\x00b", "a/../b", "./a", "a//b", "a/", "", "a\nb"],
    ids=[
        "dotdot",
        "absolute",
        "backslash",
        "nul",
        "inner-dotdot",
        "dot",
        "empty-seg",
        "trail",
        "empty",
        "newline",
    ],
)
def test_fs_traversal_and_ambiguity_refused(servers, ctx, path):
    adapter = mcp_adapter("reports", servers["reports"], {"write_report": WRITE_SCHEMA})
    with pytest.raises(RejectionError) as exc:
        adapter.normalize(raw("reports", "write_report", name=path, content="x"), ctx)
    assert reason(exc) == "invalid_arguments"


def test_fs_spaces_are_fine_and_stay_under_the_grant(servers, ctx):
    adapter = mcp_adapter("reports", servers["reports"], {"write_report": WRITE_SCHEMA})
    [interaction] = adapter.normalize(
        raw("reports", "write_report", name="q3 report.md", content="x"), ctx
    )
    assert (interaction.action, interaction.resource) == (Action.WRITE, "fs:reports/q3%20report.md")
    assert PermissionSet.parse(["write:fs:reports/*"]).allows(Action.WRITE, interaction.resource)


def test_fs_subdirectories_are_normalized_not_refused():
    assert normalized_path("2026/q3 report.md") == "2026/q3%20report.md"


# ------------------------------------------------------------------------ sql tables


@pytest.mark.parametrize(
    ("sql", "tables"),
    [
        ("SELECT COUNT(*) FROM sales.customers", ("sales.customers",)),
        (
            "select c.name, o.total from sales.customers c join sales.orders o on o.cid = c.id",
            ("sales.customers", "sales.orders"),
        ),
        ("SELECT * FROM Sales.Customers", ("sales.customers",)),  # unquoted folds to lower
        ('SELECT * FROM "sales"."customers"', ("sales.customers",)),  # quoted lower = same
        ("SELECT * FROM (SELECT * FROM sales.payments) p", ("sales.payments",)),
        (
            "SELECT count(*), sum(total), avg(total), min(total), max(total), lower(name),"
            " upper(name), length(name), coalesce(total, 0), nullif(total, 0), round(total, 2),"
            " abs(total), date_trunc('month', created), extract(year FROM created), now(),"
            " current_date, CAST(total AS int), total::text,"
            " CASE WHEN total > 1 THEN 1 ELSE 0 END FROM sales.orders",
            ("sales.orders",),
        ),
    ],
    ids=["single", "join", "case", "quoted", "subquery", "allowlisted-functions"],
)
def test_sql_tables(sql, tables):
    assert referenced_tables(sql) == tables


def test_sql_join_becomes_two_interactions(servers, ctx):
    adapter = mcp_adapter("sales_db", servers["sales_db"], {"query": QUERY_SCHEMA})
    sql = "SELECT * FROM sales.orders o JOIN sales.customers c ON o.cid = c.id"
    interactions = adapter.normalize(raw("sales_db", "query", sql=sql), ctx)
    assert [(i.action, i.resource) for i in interactions] == [
        (Action.READ, "db:sales.customers"),
        (Action.READ, "db:sales.orders"),
    ]
    assert {i.payload["arguments"]["sql"] for i in interactions} == {sql}


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM sales.orders",
        "DROP TABLE sales.orders",
        "UPDATE sales.orders SET total = 0",
        "INSERT INTO sales.orders VALUES (1)",
        "SELECT 1 FROM sales.orders; SELECT 2 FROM sales.customers",
        "WITH d AS (DELETE FROM sales.orders RETURNING *) SELECT * FROM d",
        "SELECT * INTO stolen FROM sales.customers",
        "SELECT * FROM sales.orders FOR UPDATE",
        "SELECT set_config('app.user_id', 'root@demo', true) FROM sales.customers",
        "SELECT current_setting('app.user_id')",
        "SELECT 1",
        "SELECT * FROM generate_series(1, 3)",
        "SELECT * FROM otherdb.sales.orders",
        "COPY sales.orders TO STDOUT",
        "SELECT 1 FROM sales.orders UNION SELECT 2 FROM sales.payments",
        "this is not sql (((",
        "",
        # Functions outside the allowlist; some read tables this check never sees.
        "SELECT query_to_xml($$SELECT * FROM sales.payments$$, true, false, '') "
        "FROM sales.customers",
        "SELECT query_to_xml($$SELECT set_config('app.user_id', 'root@demo', true)$$, "
        "true, false, '') FROM sales.customers",
        "SELECT pg_read_file('/etc/passwd') FROM sales.orders",
        "SELECT dblink('host=x', 'SELECT 1') FROM sales.orders",
        "SELECT made_up_function(total) FROM sales.orders",
        # CTEs (a CTE name can hide a real table in another scope).
        "WITH t AS (SELECT * FROM sales.orders) SELECT * FROM t",
        "WITH unused AS (SELECT 1) SELECT * FROM sales.customers",
        "SELECT p.relname FROM pg_class p CROSS JOIN "
        "(WITH pg_class AS (SELECT 1) SELECT * FROM sales.customers LIMIT 1) c",
        # Unqualified tables: the upstream's search_path decides what they are.
        "SELECT * FROM customers",
        # System catalogs.
        "SELECT * FROM pg_catalog.pg_class",
        "SELECT * FROM information_schema.tables",
        "SELECT * FROM pg_toast.pg_toast_1",
        # Identifiers that would collapse two tables into one resource, or are ambiguous.
        'SELECT * FROM "public.sales".customers',
        'SELECT * FROM public."sales.customers"',
        'SELECT * FROM sales."Customers"',
        'SELECT * FROM sales."cust omers"',
        'SELECT * FROM sales."a:b"',
    ],
)
def test_unsupported_sql_refused(servers, ctx, sql):
    adapter = mcp_adapter("sales_db", servers["sales_db"], {"query": QUERY_SCHEMA})
    with pytest.raises(RejectionError) as exc:
        adapter.normalize(raw("sales_db", "query", sql=sql), ctx)
    assert reason(exc) == "unsupported_sql"


def test_unqualified_tables_say_why():
    with pytest.raises(RejectionError) as exc:
        referenced_tables("SELECT * FROM customers")
    assert exc.value.message == "tables must be schema-qualified"


def test_distinct_quoted_tables_never_share_a_resource():
    """Before the identifier check both derived ``public.sales.customers``."""
    for sql in ('SELECT * FROM "public.sales".customers', 'SELECT * FROM public."sales.customers"'):
        with pytest.raises(RejectionError):
            referenced_tables(sql)


# --------------------------------------------------------------- mapping and schemas


def test_unmapped_tool_is_refused(servers, ctx):
    adapter = mcp_adapter("reports", servers["reports"], {"drop_reports": OPEN_OBJECT})
    with pytest.raises(RejectionError) as exc:
        adapter.normalize(raw("reports", "drop_reports"), ctx)
    assert reason(exc) == "tool_not_mapped"


def test_mapped_tool_the_upstream_never_advertised_is_refused(servers, ctx):
    adapter = mcp_adapter("reports", servers["reports"], {})
    with pytest.raises(RejectionError) as exc:
        adapter.normalize(raw("reports", "write_report", name="a.md", content="x"), ctx)
    assert reason(exc) == "tool_not_advertised"


@pytest.mark.parametrize(
    "arguments",
    [
        {"name": "a.md"},  # missing required
        {"name": 3, "content": "x"},  # wrong type
        {"name": "a.md", "content": "x", "path": "../../etc"},  # undeclared argument
    ],
    ids=["missing", "type", "undeclared"],
)
def test_arguments_validated_strictly_against_the_schema(servers, ctx, arguments):
    adapter = mcp_adapter("reports", servers["reports"], {"write_report": WRITE_SCHEMA})
    with pytest.raises(RejectionError) as exc:
        adapter.normalize(raw("reports", "write_report", **arguments), ctx)
    assert reason(exc) == "invalid_arguments"


def test_invalid_upstream_schema_fails_closed(servers, ctx):
    adapter = mcp_adapter("web", servers["web"], {"fetch": {"type": "objekt"}})
    with pytest.raises(RejectionError) as exc:
        adapter.normalize(raw("web", "fetch", url="https://example.com"), ctx)
    assert reason(exc) == "invalid_arguments"


def test_malformed_call_params_are_invalid(servers, ctx):
    adapter = mcp_adapter("web", servers["web"], {"fetch": FETCH_SCHEMA})
    call = RawCall(channel=Channel.MCP, server="web", data={"name": "fetch", "arguments": [1]})
    with pytest.raises(RejectionError) as exc:
        adapter.normalize(call, ctx)
    assert reason(exc) == "invalid_arguments"


# ------------------------------------------------------- annotations never authorize

READ_ONLY_HINTS = {"readOnlyHint": True, "destructiveHint": False}


def test_read_only_annotation_does_not_downgrade_a_write(servers, ctx):
    """The schema map is all the adapter ever sees of a tool; annotations cannot reach it."""
    tool_listing = {
        "name": "write_report",
        "inputSchema": WRITE_SCHEMA,
        "annotations": READ_ONLY_HINTS,
    }
    adapter = mcp_adapter(
        "reports", servers["reports"], {tool_listing["name"]: tool_listing["inputSchema"]}
    )
    [interaction] = adapter.normalize(raw("reports", "write_report", name="a.md", content="x"), ctx)
    assert interaction.action is Action.WRITE


def test_read_only_annotation_does_not_map_an_unmapped_tool(servers, ctx):
    tool_listing = {
        "name": "drop_reports",
        "inputSchema": OPEN_OBJECT,
        "annotations": READ_ONLY_HINTS,
    }
    adapter = mcp_adapter(
        "reports", servers["reports"], {tool_listing["name"]: tool_listing["inputSchema"]}
    )
    with pytest.raises(RejectionError) as exc:
        adapter.normalize(raw("reports", "drop_reports"), ctx)
    assert reason(exc) == "tool_not_mapped"
