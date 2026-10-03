"""Intent-judge session state against every store: flags never evicted silently, goal set once."""

from gateway.core.envelope import FlaggedToolCall
from gateway.core.types import SessionMode
from gateway.policy.schema import Sessions
from gateway.sessions import MAX_FLAGGED_TOOL_CALLS, SessionBinding, SessionUpdate

ANNA = SessionBinding(principal="anna@demo", actor="databot", mode=SessionMode.INTERACTIVE)
LIMITS = Sessions(idle_ttl_s=3600, max_lifetime_s=86400)


def flags(start: int, count: int) -> tuple[FlaggedToolCall, ...]:
    return tuple(
        FlaggedToolCall(tool="query", args_digest=f"{n:064x}") for n in range(start, start + count)
    )


async def flag(store, batch: tuple[FlaggedToolCall, ...]):
    return await store.apply("s-1", SessionUpdate(flagged_tool_calls=batch), half_life_s=600.0)


async def test_up_to_the_cap_nothing_overflows(store):
    await store.open("s-1", ANNA, LIMITS)
    state = await flag(store, flags(0, MAX_FLAGGED_TOOL_CALLS))
    state = await flag(store, flags(0, 3))  # the same flags again: no growth
    assert len(state.flagged_tool_calls) == MAX_FLAGGED_TOOL_CALLS
    assert state.flags_overflowed is False


async def test_overflow_is_marked_sticky_and_survives_another_store(backend):
    store = backend.store()
    await store.open("s-1", ANNA, LIMITS)
    await flag(store, flags(0, MAX_FLAGGED_TOOL_CALLS))
    state = await flag(store, flags(MAX_FLAGGED_TOOL_CALLS, 2))
    assert state.flags_overflowed is True
    assert len(state.flagged_tool_calls) == MAX_FLAGGED_TOOL_CALLS
    assert state.flagged_tool_calls[-1] == flags(MAX_FLAGGED_TOOL_CALLS + 1, 1)[0]  # newest kept
    state = await store.apply("s-1", SessionUpdate(risk_delta=0.1), half_life_s=600.0)
    assert state.flags_overflowed is True  # later calls never clear it
    reopened = await backend.store().get("s-1")  # a second gateway (Redis) / the same (memory)
    assert reopened is not None
    assert reopened.flags_overflowed is True


async def test_goal_is_set_once(store):
    await store.open("s-1", ANNA, LIMITS)
    await store.apply("s-1", SessionUpdate(goal="count customers"), half_life_s=600.0)
    state = await store.apply("s-1", SessionUpdate(goal="export payments"), half_life_s=600.0)
    assert state.goal == "count customers"
