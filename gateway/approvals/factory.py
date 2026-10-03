"""Builds the approval and kill switch stores the settings name. Never falls back to memory.

Both use the ``state`` Redis (`gateway.state_redis`: ``ACL_REDIS_URL``, ``ACL_REDIS_PASSWORD``)
and the store kind ``ACL_SESSION_STORE`` (default: the budget store's kind).
"""

from dataclasses import dataclass

from gateway.approvals.kill_switch import (
    InMemoryKillSwitchStore,
    KillSwitchStore,
    RedisKillSwitchStore,
)
from gateway.approvals.redis_store import RedisApprovalStore
from gateway.approvals.store import ApprovalStore, InMemoryApprovalStore
from gateway.settings import Settings
from gateway.state_redis import redis_client_from_settings, state_store_kind


@dataclass(frozen=True, slots=True)
class OperatorStores:
    approvals: ApprovalStore
    kills: KillSwitchStore

    async def aclose(self) -> None:
        await self.approvals.aclose()
        await self.kills.aclose()


def operator_stores_from_settings(settings: Settings) -> OperatorStores:
    match state_store_kind(settings):
        case "memory":
            return OperatorStores(InMemoryApprovalStore(), InMemoryKillSwitchStore())
        case "redis":
            client = redis_client_from_settings(settings)  # one client, both stores
            return OperatorStores(RedisApprovalStore(client), RedisKillSwitchStore(client))
