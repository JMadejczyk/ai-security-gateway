# Concepts

## The thesis

Detecting a threat should not just block one request. It should **change what the agent may
do for the rest of the session**. In an interactive session that means narrowing permissions.
For a registered autonomous system it means throttling and human approval instead, so the
business process keeps running while risky actions wait for a person.

## Two identities on every call

An agent acts on behalf of a person, but must not inherit all of that person's rights. Every
request carries a signed token with both: `sub`, the person (or `svc:<agent>` for an
autonomous job), and `act.sub`, the agent. The decision is made on the intersection:

```text
base_allowed =
      principal grants match (action, resource)     # U: the user's roles
  AND agent grants match (action, resource)          # A: the agent's allow list
  AND task scope matches (action, resource)          # T: the token's scope
  AND no explicit deny matches
```

Session risk can only **add** restrictions and obligations (approval, throttling, redaction)
on top of that; it never widens it. An approval never grants an operation outside
`base_allowed`.

Permissions are `action:resource` strings, for example `read:db:sales.payments`,
`write:fs:reports/*` or `generate:model:qwen3:8b`. The gateway also forwards a signed principal
assertion to the database, so Postgres row-level security decides which rows a person sees.

!!! example "Same question, two people"
    Anna (analyst) and Bartek (intern) both ask *"What is the total amount of all payments?"*.
    Anna gets the total. Bartek's role has no `read:db:sales.payments`, so the query never
    reaches the database.

## Credentials stay in the gateway

Real keys to databases, APIs and models live only in the gateway. The agent knows one URL. The
upstreams sit on internal networks shared only with the gateway, so the agent cannot bypass
the layer. See [Architecture](architecture.md#system-context-and-network-topology).

## Hybrid controls

Fifteen controls in two kinds: 11 deterministic rules (secrets, PII, SQL cost, egress,
signatures, budgets, ...) and 4 AI-based checks (prompt injection, tool poisoning, intent,
output). Cheap first, the LLM only when needed, and every judge fails closed. See
[The 15 controls](controls.md).

## Session risk and taint

Every verdict adds risk; injection-type detections also **taint** the session. Interactive
sessions lose `write`, `delete` and `egress`; autonomous ones are throttled and routed to the
approval queue. The table is on [The 15 controls](controls.md#what-a-detection-does-next).

## Human in the loop

When an action needs approval, the tool call returns an `approval_id`. An approval authorizes
**one exact pending operation**: it is bound to the principal, agent, session, tool, a digest of
the arguments and the policy version, and it expires. Approvers see a digest, never the
arguments, and can never approve their own session. Replaying a used approval is refused
(`approval_already_used`). See [Operations](operations.md).

## One policy, live

Everything above is configured in one `config/policy.yaml`, validated on load and hot-reloaded
without a restart. See [Policy file](policy.md).

## Budgets

Tokens, cost, tool calls and GPU time are budgeted per user, agent and session. Budget is
**reserved** atomically in Redis before each call and settled on actual usage, so concurrent
calls cannot overspend. If Redis is unavailable, budget-limited calls fail closed.
