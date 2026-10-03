"""Helpers for the budget suites (imported by name: tests/budget is not a package)."""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import httpx
from gateway_testkit import MutableClock, completion
from redis.asyncio import Redis

from gateway.budget.ledger import DAILY_TTL_S
from gateway.budget.model import BudgetScope, ScopeKind, SpendLimits
from gateway.budget.store import BudgetStore


@dataclass
class StoreUnderTest:
    """A store plus the backend-specific ways to inspect it."""

    kind: str
    store: BudgetStore
    clock: MutableClock
    redis: Redis | None = None

    async def ttl(self, key: str) -> float | None:
        if self.redis is not None:
            seconds = await self.redis.ttl(key)
            return float(seconds) if seconds >= 0 else None
        counters = self.store._counters.get(key)  # type: ignore[attr-defined]
        return (counters.expires_at - self.clock()).total_seconds() if counters else None


# --------------------------------------------------------------------- scripted LLM upstream


@dataclass
class ScriptedLLM(httpx.AsyncBaseTransport):
    """The LLM upstream: answers ``completion()``, after an optional delay or gate."""

    status: int = 200
    delay_s: float = 0.0
    gate: asyncio.Event | None = None  # when set, every call waits for it
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    bodies: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    usage: dict[str, int] | None = None  # overrides completion()'s usage

    @property
    def calls(self) -> int:
        return len(self.bodies)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        self.entered.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.status != 200:
            return httpx.Response(self.status, json={"error": "upstream broke"})
        answer = completion()
        if self.usage is not None:
            answer["usage"] = self.usage
        return httpx.Response(200, json=answer)


def scope(
    kind: ScopeKind = ScopeKind.USER,
    subject: str = "anna@demo",
    *,
    window: str = "2026-10-04",
    ttl_s: int = DAILY_TTL_S,
    **limits: int,
) -> BudgetScope:
    return BudgetScope(
        kind=kind, subject=subject, window=window, ttl_s=ttl_s, limits=SpendLimits(**limits)
    )
