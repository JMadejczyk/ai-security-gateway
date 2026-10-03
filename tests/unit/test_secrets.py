"""``secrets``: one positive per credential family, look-alikes that are not, spans, modes."""

import base64
import json
import time

import pytest

from gateway.controls.secrets import CLEAN, DETECTED, SecretScanner, SecretsControl
from gateway.core.catalog import MANDATORY_CONTROLS, control_spec
from gateway.core.envelope import Interaction, Span
from gateway.core.interfaces import ControlConfig
from gateway.core.types import Action, Channel, ControlMode, Decision, Stage
from gateway.policy.loader import PolicyLoadError

SCANNER = SecretScanner()
CONTROL = SecretsControl()
ALLOW = pytest.mark.control("secrets", "allow")
DENY = pytest.mark.control("secrets", "deny")
REDACT = pytest.mark.control("secrets", "redact")


def _b64url(document: dict[str, str]) -> str:
    return base64.urlsafe_b64encode(json.dumps(document).encode()).rstrip(b"=").decode()


# Fake credentials, assembled so no complete token sits in the source file.
AWS_KEY_ID = "AKIA" + "Z7QW3ERT5YUI2OPA"
AWS_SECRET = "Zk3m9Qw8Rt7Yp6Lk5Jh4" + "Gf3Ds2Aa1Ss0Dd9Ff8Gg"
GITHUB = "ghp_" + "aB3dE5fG7hI9jK1lM3nO5pQ7rS9tU1vW3xY5"
GITHUB_PAT = (
    "github_pat_"
    + "11ABCDE2Y0"
    + "a1B2c3D4e5_"
    + "F6g7H8i9J0k1L2m3N4o5P6q7R8s9T0u1V2w3X4y5Z6a7B8c9D0e1F2g3H4i5J6k7L8"
)
OPENAI = "sk-proj-" + "Ab3De5Fg7Hi9Jk1Lm3No5Pq7Rs9Tu1Vw3Xy5Za7Bc9De"
OPENAI_LEGACY = "sk-" + "Ab3De5Fg7Hi9Jk1Lm3No5Pq7Rs9Tu1Vw3Xy5Za7"
ANTHROPIC = "sk-ant-api03-" + "Ab3De5Fg7Hi9Jk1Lm3No5Pq7Rs9Tu1Vw3Xy5Za7Bc9De_-x"
SLACK = "xoxb-" + "123456789012-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx"
STRIPE = "sk_live_" + "4eC39HqLyjWDarjtT1zdp7dc"
GOOGLE = "AIza" + "SyD-9tSrke72PouQMnMX-a7eZSW0jkFMBWY"
JWT = f"{_b64url({'alg': 'HS256', 'typ': 'JWT'})}.{_b64url({'sub': 'anna'})}.c2lnbmF0dXJl_x-Y"
PEM = (
    "-----BEGIN RSA "
    + "PRIVATE KEY-----\nMIIEowIBAAKCAQEA\nq2v7yX9=\n-----END RSA PRIVATE KEY-----"
)


@pytest.mark.parametrize(
    ("text", "secret", "label"),
    [
        (f"export AWS_ACCESS_KEY_ID={AWS_KEY_ID}", AWS_KEY_ID, "API_KEY"),
        (f'aws_secret_access_key = "{AWS_SECRET}"', AWS_SECRET, "API_KEY"),
        (f"token {GITHUB} for CI", GITHUB, "ACCESS_TOKEN"),
        (f"token {GITHUB_PAT}", GITHUB_PAT, "ACCESS_TOKEN"),
        (f"OPENAI_API_KEY is {OPENAI}", OPENAI, "API_KEY"),
        (f"old key {OPENAI_LEGACY}", OPENAI_LEGACY, "API_KEY"),
        (f"claude: {ANTHROPIC}", ANTHROPIC, "API_KEY"),
        (f"slack bot {SLACK}", SLACK, "ACCESS_TOKEN"),
        (f"stripe {STRIPE}", STRIPE, "API_KEY"),
        (f"maps {GOOGLE}", GOOGLE, "API_KEY"),
        (f"Authorization: Bearer {JWT}", JWT, "ACCESS_TOKEN"),
        (f"key:\n{PEM}\n", PEM, "PRIVATE_KEY"),
        ("-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBg", None, "PRIVATE_KEY"),  # truncated
        ("DSN postgres://app:S3cr3tPw@db:5432/sales", "S3cr3tPw", "CONNECTION_STRING"),
        ("mongodb+srv://u:p4ssW0rd@cluster0.example.net/x", "p4ssW0rd", "CONNECTION_STRING"),
        ("redis://:hunter2hunter2@cache:6379/0", "hunter2hunter2", "CONNECTION_STRING"),
        ('db_password="Tr0ub4dor&3x"', "Tr0ub4dor&3x", "PASSWORD"),
        ('{"api_key": "abcd1234efgh5678"}', "abcd1234efgh5678", "API_KEY"),
        ("client_secret: Qm9vb2sh7yyZ", "Qm9vb2sh7yyZ", "SECRET"),
        ("ACCESS_TOKEN=correcthorsebatterystaple", "correcthorsebatterystaple", "ACCESS_TOKEN"),
    ],
)
def test_each_family_is_found(text, secret, label):
    (finding,) = SCANNER.find(text)
    if secret is not None:
        assert text[finding.start : finding.end] == secret
    else:
        assert text[finding.start :].startswith("-----BEGIN PRIVATE KEY-----")
    assert finding.kind.value == label


@pytest.mark.parametrize(
    "text",
    [
        "The password is required; please enter your password.",
        "We use sk-learn and sk-learn-compatible-estimators-and-pipelines here.",
        "request id 123e4567-e89b-12d3-a456-426614174000",
        "fixed in commit 3f786850e387550fdab836ed7e6dc881de23001b",
        "AWS docs example: AKIAIOSFODNN7EXAMPLE",
        "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "password=<your-password> api_key=${API_KEY} token=changeme secret=********",
        "DATABASE_URL=postgres://user:password@localhost/db",
        "postgres://localhost:5432/sales has no credentials",
        "password=abc123",  # too short to be worth a block
        "password: required",  # one character class, short
        "api_key=os.environ",  # a reference, not a value
        "eyJhbGciOi.notreally.ajwt",  # header is not JSON with alg
        "-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkq\n-----END PUBLIC KEY-----",
        "xoxb-not-a-real-token",  # no digits
        "Ile klientów mamy w Krakowie? Hasło do wifi jest na tablicy.",
    ],
)
def test_look_alikes_are_not_secrets(text):
    assert SCANNER.find(text) == []


def test_overlapping_rules_report_one_finding():
    text = f'aws_secret_access_key="{AWS_SECRET}"'  # also an assignment to *secret*key
    assert [(f.rule, text[f.start : f.end]) for f in SCANNER.find(text)] == [
        ("aws_secret_access_key", AWS_SECRET)
    ]


@pytest.mark.parametrize(
    "text",
    [
        "eyJ" + "a" * 200_000,  # a JWT header with no dots
        "-----BEGIN PRIVATE KEY-----" + "A" * 200_000,
        "password=" * 20_000,
        "postgres://" + "a" * 200_000,
        "sk-" + "a1B" * 60_000,
        "-----BEGIN PRIVATE KEY-----\n" * 20_000,  # many blocks, no END line
        'password="' + "\\" * 200_000 + '"',  # one long backslash run before the quote
        "postgres://" + "u" * 100_000 + ":" + "p" * 100_000,  # never reaches an @
    ],
)
def test_adversarial_input_scans_in_linear_time(text):
    started = time.perf_counter()
    SCANNER.find(text)
    assert time.perf_counter() - started < 1.0


def mcp_result(make_ctx, text: str) -> Interaction:
    return Interaction(
        session_id="s-test",
        principal="anna@demo",
        actor="databot",
        mode=make_ctx().mode,
        channel=Channel.MCP,
        action=Action.READ,
        resource="web:example.com",
        payload={"name": "fetch", "arguments": {"url": "https://example.com"}},
        result={"content": [{"type": "text", "text": text}]},
        context=make_ctx(),
    )


@REDACT
async def test_redact_masks_only_the_secret(make_ctx):
    text = "Zażółć: postgres://app:S3cr3tPw@db/sales"
    verdict = await CONTROL.evaluate(
        mcp_result(make_ctx, text), Stage.POST, ControlConfig(mode=ControlMode.REDACT)
    )
    start = text.index("S3cr3tPw")
    assert verdict.decision is Decision.REDACT
    assert verdict.redactions == (
        Span(path="/content/0/text", start=start, end=start + 8, label="CONNECTION_STRING"),
    )


@pytest.mark.parametrize(
    ("mode", "text", "decision", "reason_code", "risk"),
    [
        pytest.param(ControlMode.BLOCK, f"key {STRIPE}", Decision.BLOCK, DETECTED, 0.3, marks=DENY),
        pytest.param(ControlMode.BLOCK, "nothing to see", Decision.ALLOW, CLEAN, 0.0, marks=ALLOW),
        pytest.param(
            ControlMode.REDACT, f"key {STRIPE}", Decision.REDACT, DETECTED, 0.3, marks=REDACT
        ),
        pytest.param(ControlMode.REDACT, "nothing to see", Decision.ALLOW, CLEAN, 0.0, marks=ALLOW),
        # unresolved: most enforcing
        pytest.param(None, f"key {STRIPE}", Decision.BLOCK, DETECTED, 0.3, marks=DENY),
    ],
)
async def test_modes(make_ctx, mode, text, decision, reason_code, risk):
    verdict = await CONTROL.evaluate(
        mcp_result(make_ctx, text), Stage.POST, ControlConfig(mode=mode)
    )
    assert (verdict.decision, verdict.enforced, verdict.reason_code) == (
        decision,
        True,
        reason_code,
    )
    assert verdict.risk_delta == pytest.approx(risk)


@DENY
async def test_reason_names_kinds_never_values(make_ctx):
    verdict = await CONTROL.evaluate(
        mcp_result(make_ctx, f"{STRIPE} and {PEM}"), Stage.POST, ControlConfig()
    )
    assert verdict.reason == "API_KEY, PRIVATE_KEY"
    assert STRIPE not in str(verdict.model_dump(exclude={"redactions"}))


async def test_mandatory_control_never_runs_log_only(make_ctx, policy_doc, snapshot_from):
    assert CONTROL.mandatory
    assert "secrets" in MANDATORY_CONTROLS
    assert not control_spec("secrets").supports(ControlMode.LOG_ONLY)
    # The policy refuses it ...
    policy_doc["controls"]["secrets"] = {"mode": "log_only"}
    with pytest.raises(PolicyLoadError, match="mandatory"):
        snapshot_from(policy_doc)
    # ... a permissive profile does not downgrade it ...
    policy_doc["controls"]["secrets"] = {}
    policy_doc["profile"] = "permissive"
    assert snapshot_from(policy_doc).policy.resolved_control_mode("secrets") is ControlMode.BLOCK
    # ... and handed log_only anyway, the control refuses to produce a verdict (fail closed).
    with pytest.raises(ValueError, match="does not support"):
        await CONTROL.evaluate(
            mcp_result(make_ctx, STRIPE), Stage.POST, ControlConfig(mode=ControlMode.LOG_ONLY)
        )
