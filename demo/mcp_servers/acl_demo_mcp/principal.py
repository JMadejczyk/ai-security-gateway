"""Verification of the `X-ACL-Principal` header the gateway sends to trusted upstreams.

The header carries a short-lived HS256 JWT signed with `ACL_INTERNAL_KEY`, a key shared only
by the gateway and the trusted upstream (never the agent's bearer token, never the agent
signing key). Claims:

    iss     "ai-control-layer"          (configurable)
    aud     the upstream's audience     e.g. "mcp-postgres"
    sub     the authenticated principal e.g. "anna@demo" or "svc:nightly_etl"
    iat, exp                            exp - iat <= max_ttl_s (default 60 s)
    limits  execution limits from the gateway's policy (SQL upstreams require them):
            {"stmt_timeout_ms": int, "max_rows": int, "max_result_bytes": int}

Anything missing or invalid raises `PrincipalError`; callers must fail closed. The signature
matters even on an internal network: other services share `mcp_backend` (e.g. `mcp-files`),
and without the key they cannot make the database act as another user, nor lift the limits.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import jwt
from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError

PRINCIPAL_HEADER = "x-acl-principal"
_ALGORITHM = "HS256"
_MIN_KEY_BYTES = 32


class PrincipalError(Exception):
    """The request does not carry a valid principal assertion."""

    message: ClassVar[str] = "invalid principal assertion"

    def __init__(self) -> None:
        super().__init__(self.message)


class MissingPrincipalError(PrincipalError):
    message = "missing principal assertion"


class PrincipalLifetimeError(PrincipalError):
    message = "principal assertion lifetime too long"


class MissingLimitsError(PrincipalError):
    message = "principal assertion carries no execution limits"


class ExecutionLimits(BaseModel):
    """What the gateway's policy allows one statement: applied by the server, never negotiated."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stmt_timeout_ms: StrictInt = Field(gt=0)
    max_rows: StrictInt = Field(gt=0)
    max_result_bytes: StrictInt = Field(gt=0)


class PrincipalClaims(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    sub: str = Field(min_length=1, max_length=256, pattern=r"^[^\x00-\x1f\x7f]+$")
    iat: int
    exp: int
    limits: ExecutionLimits | None = None

    def require_limits(self) -> ExecutionLimits:
        """The signed execution limits, or `MissingLimitsError` (fail closed)."""
        if self.limits is None:
            raise MissingLimitsError
        return self.limits


class PrincipalVerifier(BaseModel):
    """Verifies principal assertions for one upstream audience."""

    model_config = ConfigDict(frozen=True)

    key: bytes = Field(min_length=_MIN_KEY_BYTES, repr=False)
    audience: str = Field(min_length=1)
    issuer: str = Field(min_length=1)
    max_ttl_s: int = Field(default=60, gt=0)

    def verify(self, headers: Mapping[str, str] | None) -> PrincipalClaims:
        """Return the claims asserted in `headers`, or raise `PrincipalError`.

        A malformed ``limits`` claim is refused here; a missing one only by callers that need
        it (`PrincipalClaims.require_limits`).
        """
        token = headers.get(PRINCIPAL_HEADER) if headers is not None else None
        if not token:
            raise MissingPrincipalError
        try:
            raw = jwt.decode(
                token,
                self.key,
                algorithms=[_ALGORITHM],
                audience=self.audience,
                issuer=self.issuer,
                options={"require": ["sub", "iat", "exp", "aud", "iss"]},
            )
            claims = PrincipalClaims.model_validate(raw)
        except (jwt.InvalidTokenError, ValidationError) as exc:
            raise PrincipalError from exc
        if claims.exp - claims.iat > self.max_ttl_s:
            raise PrincipalLifetimeError
        return claims
