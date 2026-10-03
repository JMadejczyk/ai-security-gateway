"""Helpers for the gateway HTTP suites: settings, a movable clock, raw JWTs, an app harness.

Imported by test modules (``tests/`` is on ``sys.path`` via the root conftest); the
fixtures that use it live in ``tests/conftest.py``.
"""

import base64
import io
import json
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import jwt

from gateway.container import GatewayContainer
from gateway.main import create_agent_app, create_operator_app
from gateway.settings import Settings

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT_POLICY = REPO_ROOT / "policy.yaml"
IDENTITIES = REPO_ROOT / "demo" / "identities.yaml"
# 64+ bytes, so the identity suite's HS512 probes do not trip PyJWT key-length warnings.
JWT_SECRET = "test-jwt-secret-" + "0123456789abcdef" * 4
INTERNAL_KEY = "test-internal-key-0123456789-abcdefghijklmn"
LLM_BASE = "http://ollama:11434/v1"  # policy.yaml's upstream; respx intercepts it
T0 = datetime(2026, 10, 4, 10, 0, tzinfo=UTC)


class MutableClock:
    """A clock tests can move forward."""

    def __init__(self, start: datetime = T0) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def make_settings(policy_path: Path = ROOT_POLICY, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "jwt_secret": JWT_SECRET,
        "internal_key": INTERNAL_KEY,
        "policy_path": policy_path,
        "identities_path": IDENTITIES,
        "policy_watch": False,
        "audit_path": None,
    }
    values.update(overrides)
    return Settings(**values)


def claims(clock: MutableClock, **overrides: Any) -> dict[str, Any]:
    """Valid claims for anna@demo on databot; overrides replace or (with None) drop keys."""
    now = int(clock().timestamp())
    payload: dict[str, Any] = {
        "iss": "ai-control-layer",
        "aud": "ai-control-layer",
        "sub": "anna@demo",
        "act": {"sub": "databot"},
        "roles": ["analyst"],
        "mode": "interactive",
        "session_id": "s-test-1",
        "iat": now,
        "exp": now + 600,
    }
    payload.update(overrides)
    return {k: v for k, v in payload.items() if v is not None}


def sign(payload: dict[str, Any], key: str = JWT_SECRET, algorithm: str = "HS256") -> str:
    return jwt.encode(payload, key, algorithm=algorithm)


def unsigned(payload: dict[str, Any], header: dict[str, Any], signature: bytes = b"") -> str:
    """A token with an arbitrary header, e.g. ``alg: none`` or ``RS256`` with junk."""

    def part(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    return ".".join(
        (part(json.dumps(header).encode()), part(json.dumps(payload).encode()), part(signature))
    )


def bearer(token: str) -> dict[str, str]:
    return {"authorization": f"Bearer {token}"}


def chat(model: str = "qwen3:8b", **extra: Any) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "How many customers?"}],
        **extra,
    }


def completion(
    content: str | None = "There are 40 customers.",
    *,
    model: str = "qwen3:8b",
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1_790_000_000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
    }


@dataclass
class Harness:
    container: GatewayContainer
    agent: httpx.AsyncClient
    operator: httpx.AsyncClient
    clock: MutableClock
    audit: io.StringIO
    policy_path: Path

    async def token(self, sub: str, **request: Any) -> str:
        response = await self.operator.post("/auth/demo-token", json={"sub": sub, **request})
        assert response.status_code == 200, response.text
        return response.json()["access_token"]

    def audit_entries(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.audit.getvalue().splitlines() if line]


@asynccontextmanager
async def running_gateway(tmp_path: Path, **settings: Any) -> AsyncIterator[Harness]:
    policy_path = tmp_path / "policy.yaml"
    shutil.copy(ROOT_POLICY, policy_path)
    clock = MutableClock()
    audit = io.StringIO()
    container = GatewayContainer.from_settings(
        make_settings(policy_path, **settings), clock=clock, audit_stream=audit, env={}
    )
    agent_app, operator_app = create_agent_app(container), create_operator_app(container)
    async with (
        container.running(),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=agent_app), base_url="http://agent"
        ) as agent,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=operator_app), base_url="http://operator"
        ) as operator,
    ):
        yield Harness(container, agent, operator, clock, audit, policy_path)
