"""The `ApprovalStore` contract, against the in-memory store, Redis over fakeredis (Lua) and a
real ``redis:7.2.16`` (marker ``redis``): create-or-get idempotency, the state machine, atomic
consumption and expiry."""

import asyncio
from datetime import timedelta

import pytest
from approvals_kit import StoreUnderTest

from gateway.approvals.model import (
    TRANSITIONS,
    Approval,
    ApprovalBinding,
    ApprovalConflictError,
    ApprovalDraft,
    ApprovalState,
    Transition,
)
from gateway.core.types import Action, Channel

S = ApprovalState
OPERATION = "ab" * 32
TIMEOUT = timedelta(seconds=600)
RETENTION_S = 7 * 24 * 3600


def draft(sut: StoreUnderTest, operation: str = OPERATION) -> ApprovalDraft:
    return ApprovalDraft(
        operation=operation,
        binding=ApprovalBinding(
            session_id="s-1",
            principal="svc:nightly_etl",
            agent="nightly_etl",
            channel=Channel.MCP,
            server="reports",
            tool="write_report",
            args_digest="cd" * 32,
        ),
        policy_revision="a1c9e2f04b7d",
        actions=(Action.WRITE,),
        resources=("fs:reports/nightly.md",),
        reasons=("session_requires_approval",),
        approver_roles=("ops-team",),
        created_at=sut.clock(),
    )


async def create(sut: StoreUnderTest, operation: str = OPERATION) -> tuple[Approval, bool]:
    return await sut.store.create_or_get(
        draft(sut, operation), expires_at=sut.clock() + TIMEOUT, retention_s=RETENTION_S
    )


async def move(sut: StoreUnderTest, approval_id: str, target: ApprovalState, **kw) -> Approval:
    return await sut.store.transition(approval_id, Transition(target=target, at=sut.clock(), **kw))


# Paths from pending to each state that a transition can reach directly.
PATHS: dict[ApprovalState, tuple[ApprovalState, ...]] = {
    S.PENDING: (),
    S.APPROVED: (S.APPROVED,),
    S.DENIED: (S.DENIED,),
    S.EXECUTING: (S.APPROVED, S.EXECUTING),
    S.SUCCEEDED: (S.APPROVED, S.EXECUTING, S.SUCCEEDED),
    S.FAILED: (S.APPROVED, S.EXECUTING, S.FAILED),
    S.UNCERTAIN: (S.APPROVED, S.EXECUTING, S.UNCERTAIN),
}


# ---------------------------------------------------------------- create-or-get


async def test_create_or_get_is_idempotent_under_concurrency(store_under_test: StoreUnderTest):
    results = await asyncio.gather(*(create(store_under_test) for _ in range(25)))
    assert len({record.id for record, _ in results}) == 1
    assert sum(created for _, created in results) == 1
    (record,) = await store_under_test.store.records()
    assert (record.id, record.state) == (f"apr-{OPERATION[:24]}", S.PENDING)
    assert record.binding.tool == "write_report"
    assert await store_under_test.store.pending_count() == 1


async def test_open_approval_keeps_the_slot_and_an_ended_one_frees_it(
    store_under_test: StoreUnderTest,
):
    first, _ = await create(store_under_test)
    await move(store_under_test, first.id, S.APPROVED, decided_by="olga@demo")
    again, created = await create(store_under_test)  # approved: still the operation's slot
    assert (again.id, again.state, created) == (first.id, S.APPROVED, False)

    await move(store_under_test, first.id, S.EXECUTING)
    await move(store_under_test, first.id, S.SUCCEEDED, outcome="upstream_ok")
    second, created = await create(store_under_test)  # used: the next one is a new approval
    assert created
    assert (second.id, second.state) == (f"{first.id}-2", S.PENDING)
    assert (await store_under_test.store.get(first.id)).state is S.SUCCEEDED  # type: ignore[union-attr]


async def test_other_operations_get_their_own_approval(store_under_test: StoreUnderTest):
    one, _ = await create(store_under_test)
    other, created = await create(store_under_test, "ef" * 32)
    assert created
    assert other.id != one.id
    assert await store_under_test.store.pending_count() == 2


# ------------------------------------------------------------------ state machine


TRANSITION_CASES = [
    pytest.param(source, target, id=f"{source}->{target}")
    for source in PATHS
    for target in ApprovalState
]


@pytest.mark.parametrize(("source", "target"), TRANSITION_CASES)
async def test_state_machine(store_under_test: StoreUnderTest, source, target):
    record, _ = await create(store_under_test)
    for step in PATHS[source]:
        record = await move(store_under_test, record.id, step)
    assert record.state is source
    # `expired` is reached only by the deadline, never by request.
    legal = target in TRANSITIONS[source] and target is not S.EXPIRED
    if legal:
        moved = await move(store_under_test, record.id, target)
        assert moved.state is target
    else:
        with pytest.raises(ApprovalConflictError) as caught:
            await move(store_under_test, record.id, target)
        assert caught.value.current is not None
        assert caught.value.current.state is source  # nothing changed
    expected_pending = int((target if legal else source) is S.PENDING)
    assert await store_under_test.store.pending_count() == expected_pending


async def test_decision_fields_are_recorded(store_under_test: StoreUnderTest):
    record, _ = await create(store_under_test)
    deadline = store_under_test.clock() + timedelta(seconds=900)
    approved = await move(
        store_under_test,
        record.id,
        S.APPROVED,
        decided_by="olga@demo",
        note="ok for tonight",
        expires_at=deadline,
    )
    assert (approved.decided_by, approved.note) == ("olga@demo", "ok for tonight")
    assert approved.decided_at == store_under_test.clock()
    assert abs((approved.expires_at - deadline).total_seconds()) < 0.001
    stored = await store_under_test.store.get(record.id)
    assert stored == approved


async def test_transition_of_an_unknown_id_conflicts(store_under_test: StoreUnderTest):
    with pytest.raises(ApprovalConflictError) as caught:
        await move(store_under_test, "apr-" + "0" * 24, S.APPROVED)
    assert caught.value.current is None


async def test_two_concurrent_consumers_exactly_one_wins(store_under_test: StoreUnderTest):
    record, _ = await create(store_under_test)
    await move(store_under_test, record.id, S.APPROVED)

    async def consume() -> bool:
        try:
            await move(store_under_test, record.id, S.EXECUTING)
        except ApprovalConflictError:
            return False
        return True

    outcomes = await asyncio.gather(*(consume() for _ in range(10)))
    assert outcomes.count(True) == 1
    assert (await store_under_test.store.get(record.id)).state is S.EXECUTING  # type: ignore[union-attr]


# ------------------------------------------------------------------------- expiry


async def test_a_pending_approval_expires_at_its_deadline(store_under_test: StoreUnderTest):
    record, _ = await create(store_under_test)
    store_under_test.clock.advance(599)
    assert await store_under_test.store.due(store_under_test.clock()) == []
    store_under_test.clock.advance(1)
    assert await store_under_test.store.due(store_under_test.clock()) == [record.id]

    with pytest.raises(ApprovalConflictError) as caught:  # too late to approve
        await move(store_under_test, record.id, S.APPROVED)
    assert caught.value.current is not None
    assert caught.value.current.state is S.EXPIRED
    assert caught.value.current.outcome == "approval_timeout"
    assert await store_under_test.store.due(store_under_test.clock()) == []
    assert await store_under_test.store.pending_count() == 0


async def test_an_approved_approval_expires_and_cannot_be_consumed(
    store_under_test: StoreUnderTest,
):
    record, _ = await create(store_under_test)
    await move(store_under_test, record.id, S.APPROVED)
    store_under_test.clock.advance(600)
    with pytest.raises(ApprovalConflictError) as caught:
        await move(store_under_test, record.id, S.EXECUTING)
    assert caught.value.current is not None
    assert caught.value.current.state is S.EXPIRED


async def test_expiring_by_deadline_and_then_a_fresh_approval(store_under_test: StoreUnderTest):
    record, _ = await create(store_under_test)
    store_under_test.clock.advance(601)
    expired = await move(store_under_test, record.id, S.EXPIRED)  # the sweeper's move
    assert expired.state is S.EXPIRED
    with pytest.raises(ApprovalConflictError):  # already expired: no second expiry
        await move(store_under_test, record.id, S.EXPIRED)
    fresh, created = await create(store_under_test)
    assert created
    assert fresh.id == f"{record.id}-2"


async def test_create_or_get_expires_a_stale_open_approval(store_under_test: StoreUnderTest):
    record, _ = await create(store_under_test)
    store_under_test.clock.advance(601)
    fresh, created = await create(store_under_test)
    assert created
    assert fresh.id != record.id
    assert (await store_under_test.store.get(record.id)).state is S.EXPIRED  # type: ignore[union-attr]
    assert await store_under_test.store.pending_count() == 1


async def test_listing_is_newest_first_and_filters_by_state(store_under_test: StoreUnderTest):
    old, _ = await create(store_under_test)
    store_under_test.clock.advance(1)
    new, _ = await create(store_under_test, "ef" * 32)
    await move(store_under_test, old.id, S.DENIED)
    assert [r.id for r in await store_under_test.store.records()] == [new.id, old.id]
    pending = await store_under_test.store.records(frozenset({S.PENDING}))
    assert [r.id for r in pending] == [new.id]
    assert [r.id for r in await store_under_test.store.records(limit=1)] == [new.id]
