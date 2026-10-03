"""Builds the budget store the settings name. Never falls back to memory on its own."""

from gateway.budget.redis_store import RedisBudgetStore
from gateway.budget.store import BudgetStore, InMemoryBudgetStore
from gateway.clock import Clock, utc_now
from gateway.settings import Settings


def budget_store_from_settings(settings: Settings, *, clock: Clock = utc_now) -> BudgetStore:
    """``redis`` connects lazily, so a gateway starts (and refuses budget-limited calls with
    503) while Redis is down; ``memory`` only when ``ACL_BUDGET_STORE=memory`` says so."""
    match settings.budget_store:
        case "memory":
            return InMemoryBudgetStore(clock=clock)
        case "redis":
            password = settings.redis_password
            return RedisBudgetStore.from_url(
                settings.redis_url,
                password=password.get_secret_value() if password is not None else None,
            )
