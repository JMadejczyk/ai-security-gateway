"""Which label sets of the bounded counters exist before their first event.

`gateway.telemetry.initialize_series` creates them at 0 so a dashboard's ``increase()`` sees
each one's first event. This module derives them from the policy and the known identities,
limited to what the policy can produce: a ``(user, agent)`` pair only when the agent may act
for that principal, a model only from the ``pricing`` table (any other is ``other``), a rule
only when it raises alerts.
"""

from collections.abc import Iterable, Iterator

from gateway.core.catalog import CONTROL_CATALOG
from gateway.core.types import SessionMode
from gateway.identity import principal_allowed, service_principal
from gateway.policy.loader import PolicySnapshot
from gateway.telemetry import OTHER_LABEL, initialize_series


def spend_labels(snapshot: PolicySnapshot, humans: Iterable[str]) -> Iterator[tuple[str, str, str]]:
    """``(user, agent, model)`` for every pair the policy allows, every priced model + other.

    Unknown human principals are labelled ``other`` (the recorder's bucketing), so interactive
    agents also get the ``other`` user.
    """
    policy = snapshot.policy
    models = (*policy.pricing, OTHER_LABEL)
    known = frozenset(humans)
    for agent_id, agent in policy.agents.items():
        if agent.type is SessionMode.AUTONOMOUS:
            candidates: tuple[str, ...] = (service_principal(agent_id),)
        else:
            candidates = (*sorted(known), OTHER_LABEL)
        for user in candidates:
            if user != OTHER_LABEL and not principal_allowed(user, agent_id, agent):
                continue
            for model in models:
                yield user, agent_id, model


def alert_rule_labels(snapshot: PolicySnapshot) -> Iterator[str]:
    """``<mode>.<index>`` of every risk rule with ``alert: true`` (the pipeline's label)."""
    for mode in SessionMode:
        for index, rule in enumerate(snapshot.policy.risk_rules.for_mode(mode)):
            if rule.then.alert:
                yield f"{mode.value}.{index}"


def initialize_metric_series(snapshot: PolicySnapshot, humans: Iterable[str]) -> None:
    initialize_series(
        agents=snapshot.policy.agents,
        controls=CONTROL_CATALOG,
        spend=spend_labels(snapshot, humans),
        alert_rules=alert_rule_labels(snapshot),
    )
