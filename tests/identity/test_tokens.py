"""Token verification: cryptographic checks, lifetime, and delegation against the policy."""

import jwt
import pytest
from gateway_testkit import (
    IDENTITIES,
    INTERNAL_KEY,
    JWT_SECRET,
    MutableClock,
    claims,
    running_gateway,
    sign,
    unsigned,
)

from gateway.core.types import SessionMode
from gateway.identity import (
    DemoIdentities,
    DemoTokenIssuer,
    DemoTokenRequest,
    TokenError,
    TokenReason,
    TokenVerifier,
    UnknownIdentityError,
    mint_principal_assertion,
)
from gateway.sessions import InMemorySessionStore


@pytest.fixture
def clock() -> MutableClock:
    return MutableClock()


@pytest.fixture
def verifier(clock) -> TokenVerifier:
    return TokenVerifier(JWT_SECRET.encode(), clock=clock)


@pytest.fixture
def sessions(clock) -> InMemorySessionStore:
    return InMemorySessionStore(clock=clock)


@pytest.fixture
def issuer(verifier, sessions, clock) -> DemoTokenIssuer:
    return DemoTokenIssuer(
        DemoIdentities.load(IDENTITIES),
        JWT_SECRET.encode(),
        verifier,
        sessions.is_retired,
        clock=clock,
    )


def refusal(verifier, snapshot, token: str | None) -> TokenError:
    with pytest.raises(TokenError) as caught:
        verifier.verify(token, snapshot)
    return caught.value


def test_valid_token(verifier, snapshot, clock):
    verified = verifier.verify(sign(claims(clock, scope=["generate:model:*"])), snapshot)
    assert (verified.sub, verified.agent, verified.mode) == (
        "anna@demo",
        "databot",
        SessionMode.INTERACTIVE,
    )
    assert verified.scope is not None
    assert verified.scope.as_strings() == ["generate:model:*"]
    assert verified.principal_context().task_scope == verified.scope


def test_missing_token_is_401(verifier, snapshot):
    error = refusal(verifier, snapshot, None)
    assert (error.reason, error.status_code) == (TokenReason.MISSING, 401)


def test_forged_signature(verifier, snapshot, clock):
    forged = sign(claims(clock), key="another-secret-of-the-same-length-0123456789")
    assert refusal(verifier, snapshot, forged).reason is TokenReason.INVALID


@pytest.mark.parametrize(
    ("header", "signature"),
    [
        ({"alg": "none", "typ": "JWT"}, b""),
        ({"alg": "RS256", "typ": "JWT"}, b"not-an-rsa-signature"),
        ({"alg": "HS512", "typ": "JWT"}, b"x" * 64),
    ],
    ids=["none", "RS256", "HS512"],
)
def test_algorithm_is_pinned(verifier, snapshot, clock, header, signature):
    token = unsigned(claims(clock), header, signature)
    error = refusal(verifier, snapshot, token)
    assert (error.reason, error.status_code) == (TokenReason.ALG_NOT_ALLOWED, 401)


def test_hs512_with_the_right_key_is_still_refused(verifier, snapshot, clock):
    token = sign(claims(clock), algorithm="HS512")
    assert refusal(verifier, snapshot, token).reason is TokenReason.ALG_NOT_ALLOWED


def test_expired(verifier, snapshot, clock):
    now = int(clock().timestamp())
    token = sign(claims(clock, iat=now - 600, exp=now - 1))
    assert refusal(verifier, snapshot, token).reason is TokenReason.EXPIRED


def test_expires_with_the_clock(verifier, snapshot, clock):
    token = sign(claims(clock))
    verifier.verify(token, snapshot)
    clock.advance(600)
    assert refusal(verifier, snapshot, token).reason is TokenReason.EXPIRED


def test_issued_in_the_future(verifier, snapshot, clock):
    now = int(clock().timestamp())
    token = sign(claims(clock, iat=now + 60, exp=now + 600))
    assert refusal(verifier, snapshot, token).reason is TokenReason.INVALID


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"aud": "someone-else"}, TokenReason.WRONG_AUDIENCE),
        ({"iss": "evil-issuer"}, TokenReason.WRONG_ISSUER),
        ({"aud": None}, TokenReason.INVALID),
        ({"session_id": None}, TokenReason.INVALID),
        ({"session_id": "has spaces"}, TokenReason.INVALID),
        ({"scope": ["frobnicate:db:x"]}, TokenReason.INVALID),
        ({"admin": True}, TokenReason.INVALID),  # unknown claims are refused
        ({"exp": "soon"}, TokenReason.INVALID),
    ],
    ids=["aud", "iss", "no-aud", "no-session", "bad-session", "bad-scope", "extra", "exp-type"],
)
def test_claim_checks(verifier, snapshot, clock, override, reason):
    assert refusal(verifier, snapshot, sign(claims(clock, **override))).reason is reason


def test_lifetime_over_one_hour(verifier, snapshot, clock):
    now = int(clock().timestamp())
    token = sign(claims(clock, iat=now, exp=now + 3601))
    error = refusal(verifier, snapshot, token)
    assert (error.reason, error.status_code) == (TokenReason.LIFETIME_EXCEEDED, 401)


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"act": {"sub": "ghost_agent"}}, TokenReason.UNKNOWN_AGENT),
        ({"mode": "autonomous"}, TokenReason.MODE_MISMATCH),
        (
            {"sub": "svc:nightly_etl", "act": {"sub": "nightly_etl"}, "roles": []},
            TokenReason.MODE_MISMATCH,  # interactive token for the autonomous agent
        ),
        ({"roles": ["analyst", "superuser"]}, TokenReason.UNKNOWN_ROLE),
        ({"sub": "svc:nightly_etl", "roles": []}, TokenReason.PRINCIPAL_NOT_ALLOWED),
        (
            {"act": {"sub": "nightly_etl"}, "mode": "autonomous"},
            TokenReason.PRINCIPAL_NOT_ALLOWED,  # a human cannot drive the autonomous agent
        ),
    ],
    ids=[
        "unknown-agent",
        "autonomous-token-for-databot",
        "interactive-token-for-nightly_etl",
        "unknown-role",
        "svc-on-interactive",
        "human-on-autonomous",
    ],
)
def test_delegation_checks(verifier, snapshot, clock, override, reason):
    error = refusal(verifier, snapshot, sign(claims(clock, **override)))
    assert (error.reason, error.status_code) == (reason, 403)


def test_autonomous_service_principal_is_accepted(verifier, snapshot, clock):
    token = sign(
        claims(
            clock, sub="svc:nightly_etl", act={"sub": "nightly_etl"}, roles=[], mode="autonomous"
        )
    )
    assert verifier.verify(token, snapshot).mode is SessionMode.AUTONOMOUS


def test_agent_principals_list_is_enforced(verifier, snapshot_from, policy_doc, clock):
    policy_doc["agents"]["databot"]["principals"] = ["bartek@demo"]
    restricted = snapshot_from(policy_doc)
    assert refusal(verifier, restricted, sign(claims(clock))).reason is (
        TokenReason.PRINCIPAL_NOT_ALLOWED
    )
    bartek = sign(claims(clock, sub="bartek@demo", roles=["intern"]))
    assert verifier.verify(bartek, restricted).sub == "bartek@demo"


# --------------------------------------------------------------------------- demo issuer


async def test_demo_issuer_issues_verifiable_tokens_with_fresh_sessions(issuer, verifier, snapshot):
    first = await issuer.issue(DemoTokenRequest(sub="anna@demo"), snapshot)
    second = await issuer.issue(DemoTokenRequest(sub="anna@demo"), snapshot)
    assert first.session_id != second.session_id
    verified = verifier.verify(first.access_token, snapshot)
    assert (verified.sub, verified.agent, verified.roles) == ("anna@demo", "databot", ("analyst",))


async def test_demo_issuer_keeps_a_requested_session(issuer, snapshot):
    issued = await issuer.issue(
        DemoTokenRequest(sub="svc:nightly_etl", session_id="s-etl-1"), snapshot
    )
    assert (issued.session_id, issued.agent, issued.mode) == (
        "s-etl-1",
        "nightly_etl",
        SessionMode.AUTONOMOUS,
    )


async def test_demo_issuer_refuses_unknown_identity(issuer, snapshot):
    with pytest.raises(UnknownIdentityError):
        await issuer.issue(DemoTokenRequest(sub="mallory@demo"), snapshot)


async def test_demo_issuer_refuses_a_delegation_the_policy_rejects(issuer, snapshot):
    with pytest.raises(TokenError) as caught:
        await issuer.issue(DemoTokenRequest(sub="anna@demo", agent="nightly_etl"), snapshot)
    assert caught.value.reason is TokenReason.MODE_MISMATCH


def test_demo_token_lifetime_is_capped():
    with pytest.raises(ValueError, match="ttl_s"):
        DemoTokenRequest(sub="anna@demo", ttl_s=7200)


async def test_demo_endpoint_404_when_disabled(tmp_path):
    async with running_gateway(tmp_path, demo_tokens=False) as gw:
        response = await gw.operator.post("/auth/demo-token", json={"sub": "root@demo"})
    assert response.status_code == 404


async def test_demo_endpoint_refuses_unknown_identity(gateway):
    response = await gateway.operator.post("/auth/demo-token", json={"sub": "mallory@demo"})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "unknown_identity"


def test_principal_assertion_matches_the_mcp_contract(clock):
    token = mint_principal_assertion(
        INTERNAL_KEY.encode(), audience="mcp-postgres", principal="anna@demo", clock=clock
    )
    decoded = jwt.decode(
        token,
        INTERNAL_KEY,
        algorithms=["HS256"],
        audience="mcp-postgres",
        issuer="ai-control-layer",
        options={"verify_exp": False, "verify_iat": False},
    )
    assert decoded["sub"] == "anna@demo"
    assert decoded["exp"] - decoded["iat"] == 60
    with pytest.raises(ValueError, match="at most 60"):
        mint_principal_assertion(INTERNAL_KEY.encode(), audience="x", principal="a", ttl_s=61)
