"""Verification of the `X-ACL-Principal` header the gateway sends to trusted upstreams.

The header carries a short-lived HS256 JWT signed with `ACL_INTERNAL_KEY`, a key shared only
by the gateway and the trusted upstream (never the agent's bearer token, never the agent
signing key). Required claims:

    iss  "ai-control-layer"          (configurable)
    aud  the upstream's audience     e.g. "mcp-postgres"
    sub  the authenticated principal e.g. "anna@demo" or "svc:nightly_etl"
    iat, exp                         exp - iat <= max_ttl_s (default 60 s)

Anything missing or invalid raises `PrincipalError`; callers must fail closed. The signature
matters even on an internal network: other services share `mcp_backend` (e.g. `mcp-files`),
and without the key they cannot make the database act as another user.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import jwt
from pydantic import BaseModel, ConfigDict, Field, ValidationError

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


class PrincipalClaims(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    sub: str = Field(min_length=1, max_length=256, pattern=r"^[^\x00-\x1f\x7f]+$")
    iat: int
    exp: int


class PrincipalVerifier(BaseModel):
    """Verifies principal assertions for one upstream audience."""

    model_config = ConfigDict(frozen=True)

    key: bytes = Field(min_length=_MIN_KEY_BYTES, repr=False)
    audience: str = Field(min_length=1)
    issuer: str = Field(min_length=1)
    max_ttl_s: int = Field(default=60, gt=0)

    def verify(self, headers: Mapping[str, str] | None) -> str:
        """Return the principal asserted in `headers`, or raise `PrincipalError`."""
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
        return claims.sub
