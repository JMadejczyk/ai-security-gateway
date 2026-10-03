"""Identity: bearer JWT verification, delegation checks and the demo token issuer.

SPEC "Identity and permission model" → "Identity". A token is accepted only when

- its header names the pinned algorithm (``HS256`` for the demo key) and the signature holds;
- ``iss``, ``aud``, ``exp``, ``iat`` are present and valid and its lifetime is at most 1 h;
- against the current policy snapshot: ``act.sub`` is a registered agent, ``mode`` equals that
  agent's ``type``, every role exists, and the principal may be represented by that agent.

Every refusal is a `TokenError` with a structured reason code.
"""

import secrets
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Final, Literal, Self

import jwt
import yaml
from pydantic import Field, StrictInt, StringConstraints, ValidationError

from gateway.clock import Clock, utc_now
from gateway.core.envelope import FrozenModel
from gateway.core.types import SessionMode
from gateway.errors import RejectionError
from gateway.policy.evaluator import PrincipalContext
from gateway.policy.loader import PolicySnapshot
from gateway.policy.permissions import PermissionSet
from gateway.policy.schema import SERVICE_PRINCIPAL_PREFIX, Agent, Policy, service_principal

ISSUER: Final = "ai-control-layer"
AUDIENCE: Final = "ai-control-layer"
ALGORITHM: Final = "HS256"
MAX_TOKEN_LIFETIME_S: Final = 3600
PRINCIPAL_ASSERTION_TTL_S: Final = 60

type Subject = Annotated[
    str, StringConstraints(min_length=1, max_length=256, pattern=r"^[^\s\x00-\x1f\x7f]+$")
]
type SessionId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]


class TokenReason(StrEnum):
    MISSING = "token_missing"
    INVALID = "token_invalid"
    EXPIRED = "token_expired"
    WRONG_AUDIENCE = "wrong_audience"
    WRONG_ISSUER = "wrong_issuer"
    ALG_NOT_ALLOWED = "alg_not_allowed"
    LIFETIME_EXCEEDED = "lifetime_exceeded"
    UNKNOWN_AGENT = "unknown_agent"
    MODE_MISMATCH = "mode_mismatch"
    UNKNOWN_ROLE = "unknown_role"
    PRINCIPAL_NOT_ALLOWED = "principal_not_allowed"


# The credential itself is bad: 401. A sound credential asserting a delegation the policy
# does not register is a permission problem: 403.
_CREDENTIAL_FAILURES: Final = frozenset(
    {
        TokenReason.MISSING,
        TokenReason.INVALID,
        TokenReason.EXPIRED,
        TokenReason.WRONG_AUDIENCE,
        TokenReason.WRONG_ISSUER,
        TokenReason.ALG_NOT_ALLOWED,
        TokenReason.LIFETIME_EXCEEDED,
    }
)


class TokenError(RejectionError):
    """A bearer token was refused."""

    def __init__(self, reason: TokenReason) -> None:
        super().__init__(reason.value, f"token refused: {reason.value.replace('_', ' ')}")
        self.reason = reason
        self.status_code = 401 if reason in _CREDENTIAL_FAILURES else 403


class Actor(FrozenModel):
    """RFC 8693 ``act`` claim: the agent acting on the principal's behalf."""

    sub: str = Field(min_length=1)


class TokenClaims(FrozenModel):
    """The claims of an agent or operator token (unknown claims are rejected)."""

    iss: str
    aud: str
    sub: Subject  # the principal: a human, or svc:<agent> for an autonomous agent
    act: Actor
    roles: tuple[str, ...] = ()
    mode: SessionMode
    session_id: SessionId
    scope: PermissionSet | None = None  # task scope: absent = unrestricted, [] = nothing
    iat: StrictInt
    exp: StrictInt

    @property
    def agent(self) -> str:
        return self.act.sub

    def principal_context(self) -> PrincipalContext:
        return PrincipalContext(
            principal=self.sub,
            roles=self.roles,
            agent=self.agent,
            mode=self.mode,
            task_scope=self.scope,
        )


def _sign(claims: dict[str, Any], key: bytes) -> str:
    # PyJWT annotates `key` with a union over optional `cryptography` types; without that
    # package installed pyright sees part of it as Unknown. HS256 only ever takes bytes here.
    return jwt.encode(claims, key, algorithm=ALGORITHM)  # pyright: ignore[reportUnknownMemberType]


def encode_token(claims: TokenClaims, key: bytes) -> str:
    return _sign(claims.model_dump(mode="json", exclude_none=True), key)


def principal_allowed(principal: str, agent_id: str, agent: Agent) -> bool:
    """Delegation rule: may ``agent`` represent ``principal``?

    Autonomous agents act only as their own service principal; interactive agents act for
    humans, never a service principal. An explicit ``principals`` list applies to both types
    (``principals: []`` means nobody), mirroring the evaluator's per-call check.
    """
    if agent.type is SessionMode.AUTONOMOUS:
        if principal != service_principal(agent_id):
            return False
    elif principal.startswith(SERVICE_PRINCIPAL_PREFIX):
        return False
    return agent.principals is None or principal in agent.principals


def delegation_refusal(policy: Policy, claims: TokenClaims) -> TokenReason | None:
    """Registration checks of a cryptographically valid token against one policy."""
    agent = policy.agents.get(claims.agent)
    if agent is None:
        return TokenReason.UNKNOWN_AGENT
    if claims.mode is not agent.type:
        return TokenReason.MODE_MISMATCH
    if any(role not in policy.roles for role in claims.roles):
        return TokenReason.UNKNOWN_ROLE
    if not principal_allowed(claims.sub, claims.agent, agent):
        return TokenReason.PRINCIPAL_NOT_ALLOWED
    return None


class TokenVerifier:
    """Verifies bearer tokens; time checks use the injected clock."""

    def __init__(
        self,
        key: bytes,
        *,
        clock: Clock = utc_now,
        max_lifetime_s: int = MAX_TOKEN_LIFETIME_S,
    ) -> None:
        self._key = key
        self._clock = clock
        self._max_lifetime_s = max_lifetime_s

    def verify(self, token: str | None, snapshot: PolicySnapshot) -> TokenClaims:
        """Return the claims of ``token`` or raise `TokenError`."""
        if not token:
            raise TokenError(TokenReason.MISSING)
        claims = self._decode(token)
        self._check_lifetime(claims)
        if refusal := delegation_refusal(snapshot.policy, claims):
            raise TokenError(refusal)
        return claims

    def _decode(self, token: str) -> TokenClaims:
        try:
            header = jwt.get_unverified_header(token)
        except jwt.InvalidTokenError:
            raise TokenError(TokenReason.INVALID) from None
        if header.get("alg") != ALGORITHM:
            raise TokenError(TokenReason.ALG_NOT_ALLOWED)
        try:
            raw = jwt.decode(  # pyright: ignore[reportUnknownMemberType] -- see _sign
                token,
                self._key,
                algorithms=[ALGORITHM],
                audience=AUDIENCE,
                issuer=ISSUER,
                # Times are checked against the injected clock in _check_lifetime.
                options={
                    "require": ["iss", "aud", "sub", "iat", "exp"],
                    "verify_exp": False,
                    "verify_iat": False,
                    "verify_nbf": False,
                },
            )
        except jwt.InvalidAudienceError:
            raise TokenError(TokenReason.WRONG_AUDIENCE) from None
        except jwt.InvalidIssuerError:
            raise TokenError(TokenReason.WRONG_ISSUER) from None
        except jwt.InvalidAlgorithmError:
            raise TokenError(TokenReason.ALG_NOT_ALLOWED) from None
        except jwt.InvalidTokenError:
            raise TokenError(TokenReason.INVALID) from None
        try:
            return TokenClaims.model_validate(raw)
        except ValidationError:
            raise TokenError(TokenReason.INVALID) from None

    def _check_lifetime(self, claims: TokenClaims) -> None:
        now = int(self._clock().timestamp())
        if claims.exp <= claims.iat or claims.iat > now:
            raise TokenError(TokenReason.INVALID)
        if claims.exp - claims.iat > self._max_lifetime_s:
            raise TokenError(TokenReason.LIFETIME_EXCEEDED)
        if claims.exp <= now:
            raise TokenError(TokenReason.EXPIRED)


# ------------------------------------------------------------------------- demo issuer


class DemoIdentity(FrozenModel):
    """One entry of `demo/identities.yaml`."""

    roles: tuple[str, ...] = ()
    mode: SessionMode
    default_agent: str = Field(min_length=1)
    description: str = ""


class DemoIdentities(FrozenModel):
    identities: dict[Subject, DemoIdentity]

    @classmethod
    def load(cls, path: Path) -> Self:
        document: object = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(document)

    def subjects(self) -> frozenset[str]:
        return frozenset(self.identities)


class DemoTokenRequest(FrozenModel):
    """Body of ``POST /auth/demo-token``; roles and mode always come from the identities file."""

    sub: Subject
    agent: str | None = Field(default=None, min_length=1)  # default: the identity's agent
    session_id: SessionId | None = None  # default: a fresh session
    scope: PermissionSet | None = None
    ttl_s: int = Field(default=MAX_TOKEN_LIFETIME_S, gt=0, le=MAX_TOKEN_LIFETIME_S)


class IssuedToken(FrozenModel):
    access_token: str
    token_type: Literal["Bearer"] = Field(default="Bearer")
    expires_in: int
    session_id: str
    sub: str
    agent: str
    mode: SessionMode


class UnknownIdentityError(RejectionError):
    def __init__(self) -> None:
        super().__init__("unknown_identity", "no demo identity with that subject")


def new_session_id() -> str:
    return f"s-{secrets.token_urlsafe(12)}"


class DemoTokenIssuer:
    """Issues tokens for the predefined demo identities only."""

    def __init__(
        self,
        identities: DemoIdentities,
        key: bytes,
        verifier: TokenVerifier,
        *,
        clock: Clock = utc_now,
    ) -> None:
        self._identities = identities
        self._key = key
        self._verifier = verifier
        self._clock = clock

    def issue(self, request: DemoTokenRequest, snapshot: PolicySnapshot) -> IssuedToken:
        identity = self._identities.identities.get(request.sub)
        if identity is None:
            raise UnknownIdentityError
        now = self._clock()
        claims = TokenClaims(
            iss=ISSUER,
            aud=AUDIENCE,
            sub=request.sub,
            act=Actor(sub=request.agent or identity.default_agent),
            roles=identity.roles,
            mode=identity.mode,
            session_id=request.session_id or new_session_id(),
            scope=request.scope,
            iat=int(now.timestamp()),
            exp=int((now + timedelta(seconds=request.ttl_s)).timestamp()),
        )
        token = encode_token(claims, self._key)
        # Never hand out a token the gateway itself would refuse under this policy.
        self._verifier.verify(token, snapshot)
        return IssuedToken(
            access_token=token,
            expires_in=request.ttl_s,
            session_id=claims.session_id,
            sub=claims.sub,
            agent=claims.agent,
            mode=claims.mode,
        )


def mint_principal_assertion(
    key: bytes,
    *,
    audience: str,
    principal: str,
    clock: Clock = utc_now,
    ttl_s: int = PRINCIPAL_ASSERTION_TTL_S,
) -> str:
    """``X-ACL-Principal`` for a trusted upstream: HS256 signed with ``ACL_INTERNAL_KEY``."""
    if not 0 < ttl_s <= PRINCIPAL_ASSERTION_TTL_S:
        msg = f"principal assertions live at most {PRINCIPAL_ASSERTION_TTL_S} s"
        raise ValueError(msg)
    now = int(clock().timestamp())
    claims = {"iss": ISSUER, "aud": audience, "sub": principal, "iat": now, "exp": now + ttl_s}
    return _sign(claims, key)
