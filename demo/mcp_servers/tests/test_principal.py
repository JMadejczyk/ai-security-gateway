from __future__ import annotations

import time
from typing import Any

import jwt
import pytest

from acl_demo_mcp.principal import (
    PRINCIPAL_HEADER,
    ExecutionLimits,
    MissingLimitsError,
    PrincipalError,
    PrincipalVerifier,
)

KEY = b"k" * 32
VERIFIER = PrincipalVerifier(key=KEY, audience="mcp-postgres", issuer="ai-control-layer")
LIMITS = {"stmt_timeout_ms": 3000, "max_rows": 500, "max_result_bytes": 1_048_576}


def _token(key: bytes = KEY, algorithm: str = "HS256", **overrides: Any) -> str:
    now = int(time.time())
    claims: dict[str, Any] = {
        "iss": "ai-control-layer",
        "aud": "mcp-postgres",
        "sub": "anna@demo",
        "iat": now,
        "exp": now + 30,
        "limits": LIMITS,
    }
    claims.update(overrides)
    return jwt.encode({k: v for k, v in claims.items() if v is not None}, key, algorithm=algorithm)


def test_returns_principal_and_limits_from_valid_assertion() -> None:
    claims = VERIFIER.verify({PRINCIPAL_HEADER: _token()})
    assert claims.sub == "anna@demo"
    assert claims.require_limits() == ExecutionLimits(**LIMITS)


def test_missing_header_fails_closed() -> None:
    with pytest.raises(PrincipalError):
        VERIFIER.verify({})
    with pytest.raises(PrincipalError):
        VERIFIER.verify(None)


def test_missing_limits_fail_closed_where_required() -> None:
    claims = VERIFIER.verify({PRINCIPAL_HEADER: _token(limits=None)})
    with pytest.raises(MissingLimitsError):
        claims.require_limits()


@pytest.mark.parametrize(
    "token",
    [
        _token(key=b"x" * 32),
        _token(aud="ai-control-layer"),
        _token(iss="someone-else"),
        _token(exp=int(time.time()) - 10),
        _token(sub=None),
        _token(sub=""),
        _token(exp=int(time.time()) + 3600),
        _token(algorithm="HS512"),
        "not-a-jwt",
        _token(limits={**LIMITS, "max_rows": 0}),
        _token(limits={**LIMITS, "stmt_timeout_ms": -1}),
        _token(limits={**LIMITS, "max_rows": "500"}),
        _token(limits={"stmt_timeout_ms": 3000, "max_rows": 500}),
        _token(limits={**LIMITS, "unlimited": True}),
        _token(limits="none"),
    ],
    ids=[
        "wrong-key",
        "agent-audience",
        "wrong-issuer",
        "expired",
        "no-sub",
        "empty-sub",
        "lifetime-too-long",
        "wrong-algorithm",
        "garbage",
        "zero-rows",
        "negative-timeout",
        "string-rows",
        "missing-byte-cap",
        "unknown-limit",
        "limits-not-an-object",
    ],
)
def test_rejects_invalid_assertions(token: str) -> None:
    with pytest.raises(PrincipalError):
        VERIFIER.verify({PRINCIPAL_HEADER: token})
