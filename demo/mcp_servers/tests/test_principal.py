from __future__ import annotations

import time
from typing import Any

import jwt
import pytest

from acl_demo_mcp.principal import PRINCIPAL_HEADER, PrincipalError, PrincipalVerifier

KEY = b"k" * 32
VERIFIER = PrincipalVerifier(key=KEY, audience="mcp-postgres", issuer="ai-control-layer")


def _token(key: bytes = KEY, algorithm: str = "HS256", **overrides: Any) -> str:
    now = int(time.time())
    claims: dict[str, Any] = {
        "iss": "ai-control-layer",
        "aud": "mcp-postgres",
        "sub": "anna@demo",
        "iat": now,
        "exp": now + 30,
    }
    claims.update(overrides)
    return jwt.encode({k: v for k, v in claims.items() if v is not None}, key, algorithm=algorithm)


def test_returns_principal_from_valid_assertion() -> None:
    assert VERIFIER.verify({PRINCIPAL_HEADER: _token()}) == "anna@demo"


def test_missing_header_fails_closed() -> None:
    with pytest.raises(PrincipalError):
        VERIFIER.verify({})
    with pytest.raises(PrincipalError):
        VERIFIER.verify(None)


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
    ],
)
def test_rejects_invalid_assertions(token: str) -> None:
    with pytest.raises(PrincipalError):
        VERIFIER.verify({PRINCIPAL_HEADER: token})
