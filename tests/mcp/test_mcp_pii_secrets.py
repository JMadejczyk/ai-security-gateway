"""``pii`` and ``secrets`` on ``tools/call``: arguments before the upstream, results after it.

A fetched page carrying a PESEL reaches the agent masked; one carrying a credential is
withheld (and still taints: the untrusted content reached the gateway). A credential in a
tool argument never reaches the upstream; personal data in one is masked first. No audit
line carries any of it.
"""

import pytest
from mcp_harness import MCPStack, connect, connect_all, error_text

PII_ALLOW = pytest.mark.control("pii", "allow")
PII_REDACT = pytest.mark.control("pii", "redact")
SECRETS_DENY = pytest.mark.control("secrets", "deny")

ANNA = "anna@demo"
PESEL = "44051401359"
GITHUB = "ghp_" + "aB3dE5fG7hI9jK1lM3nO5pQ7rS9tU1vW3xY5"  # fake, assembled


def assert_nothing_leaked(stack: MCPStack, *values: str) -> None:
    raw = stack.gateway.audit.getvalue()
    for value in values:
        assert value not in raw


def text_of(result: dict) -> str:
    assert result["isError"] is False, result
    return result["content"][0]["text"]


@PII_REDACT
async def test_fetched_pesel_comes_back_redacted(stack: MCPStack):
    stack.log.fetch_page = f"<p>Kontakt: Jan Kowalski, PESEL {PESEL}</p>"
    web = await connect(stack, ANNA, "web")
    result = await web.call("fetch", url="https://example.com/kontakt")
    assert text_of(result) == "<p>Kontakt: Jan Kowalski, PESEL [REDACTED:PL_PESEL]</p>"
    assert PESEL not in str(result)  # structuredContent is masked too
    entry = stack.gateway.audit_entries()[-1]
    assert entry["decision"] == "redact"
    assert {"control": "pii", "stage": "post", "decision": "redact", "enforced": True,
            "reason_code": "pii_detected"} in entry["verdicts"]  # fmt: skip
    assert_nothing_leaked(stack, PESEL)


@SECRETS_DENY
async def test_fetched_credential_is_withheld_and_still_taints(stack: MCPStack):
    stack.log.fetch_page = f"<pre>GITHUB_TOKEN={GITHUB}</pre>"
    web = await connect(stack, ANNA, "web")
    result = await web.call("fetch", url="https://example.com/leak")
    assert error_text(result) == "secret_detected"
    assert GITHUB not in str(result)
    assert len(stack.log.of("fetch")) == 1
    entry = stack.gateway.audit_entries()[-1]
    assert (entry["decision"], entry["taint"]) == ("block", True)
    assert_nothing_leaked(stack, GITHUB)


@SECRETS_DENY
async def test_credential_in_an_argument_never_reaches_the_upstream(stack: MCPStack):
    reports = await connect(stack, ANNA, "reports")
    result = await reports.call("write_report", name="q3.txt", content=f"token {GITHUB}")
    assert error_text(result) == "secret_detected"
    assert stack.log.of("write_report") == []
    assert_nothing_leaked(stack, GITHUB)


@PII_REDACT
async def test_personal_data_in_an_argument_is_masked_before_the_upstream(stack: MCPStack):
    reports = await connect(stack, ANNA, "reports")
    result = await reports.call(
        "write_report", name="q3.txt", content=f"Klient {PESEL}, tel. +48 600 700 800"
    )
    assert result["isError"] is False, result
    (call,) = stack.log.of("write_report")
    assert call.arguments["content"] == "Klient [REDACTED:PL_PESEL], tel. [REDACTED:PHONE_NUMBER]"
    assert_nothing_leaked(stack, PESEL, "600 700 800")


@pytest.mark.parametrize(
    ("page", "expected"),
    [
        pytest.param("Quarterly outlook: stable.", "Quarterly outlook: stable.", marks=PII_ALLOW),
        pytest.param(
            "Write to biuro@firma.pl", "Write to [REDACTED:EMAIL_ADDRESS]", marks=PII_REDACT
        ),
    ],
)
async def test_clean_and_dirty_results_side_by_side(stack: MCPStack, page, expected):
    stack.log.fetch_page = page
    (web,) = await connect_all(stack, ANNA, "web")
    assert text_of(await web.call("fetch", url="https://example.com/")) == expected


@SECRETS_DENY
async def test_a_quoted_password_with_spaces_is_blocked_whole(stack: MCPStack):
    reports = await connect(stack, ANNA, "reports")
    content = 'config: password="Abc12345 secret-tail"'
    result = await reports.call("write_report", name="cfg.txt", content=content)
    assert error_text(result) == "secret_detected"
    assert stack.log.of("write_report") == []
    assert_nothing_leaked(stack, "secret-tail")


@PII_REDACT
async def test_full_width_pesel_in_a_fetched_page_is_masked(stack: MCPStack):
    disguised = "".join(chr(ord(c) + 0xFEE0) for c in PESEL)
    stack.log.fetch_page = f"PESEL: {disguised}."
    web = await connect(stack, ANNA, "web")
    result = await web.call("fetch", url="https://example.com/kontakt")
    assert text_of(result) == "PESEL: [REDACTED:PL_PESEL]."
    assert disguised not in str(result)
