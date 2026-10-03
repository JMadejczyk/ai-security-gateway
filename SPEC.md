# AI Control Layer — tech spec

Oct 3, 2026 · @Jakub

## Goal and thesis

We are building a security layer that all agent traffic passes through, both to models and to tools. It is not an LLM router. It sits in front of any OpenAI-compatible gateway (LiteLLM, Ollama, OpenRouter) and in front of MCP servers.

Thesis: detecting a threat should not just block one request. It should change what the agent may do for the rest of the session. In an interactive session that means narrowing permissions. For a registered autonomous system it means throttling and human approval instead, because cutting it off could stop a business process. Effective permissions are always user permissions ∩ agent permissions ∩ task scope, reduced by session risk.

The problem raised by the challenge owner: an agent acts on behalf of a specific person but must not inherit all of that person's rights, and the same agent triggered by two people must see different data.

## Scope

In 22 hours we ship two entry points (LLM and MCP), a shared policy core with hot reload, a dozen-plus controls, Grafana and a test suite. Everything else is an extension point shown on the diagram.

| Area | In scope | Out of scope (extension point) |
| --- | --- | --- |
| Entry points | LLM proxy (OpenAI-compatible), MCP proxy (streamable HTTP) | A2A proxy, in-process SDK/library |
| Identity | Our own JWT with `sub` (user) and `act` (agent), signed with a demo key | Real IdP, full OAuth Token Exchange |
| Policies | One YAML file, validation, hot reload without restart | Policy editing UI, versioning in a database |
| Adapters | Generic MCP, SQL (Postgres), HTTP egress, filesystem, LLM | More systems (Jira, S3, Slack) |
| Controls | See Control catalog | Training our own classifiers |
| Reporting | Prometheus + Loki + Grafana, JSONL audit log export | SIEM integration |
| Upstream | Ollama directly; optionally LiteLLM in between | Load balancing, fallbacks (a router's job) |

## Architecture

One service with two entry points and a shared core. The agent knows only the gateway address; the real credentials for models and data live in the gateway.

&#91;embedded content: architecture · two entry points, shared core\]

The LLM proxy sees intent (a `tool_call` in the model's response); the MCP proxy enforces the decision when the tool runs. Both share the same session state, so taint from a tool result narrows rights on the next turn. The LLM upstream is any OpenAI-compatible URL, so the layer sits in front of an existing router instead of replacing it. A2A would be a third entry point into the same core, out of scope for the hackathon.

**No bypassing the gateway.** Upstreams accept traffic only from the gateway. Ollama, the router and MCP servers live on an internal Docker network with no published ports, and the router key (e.g. the LiteLLM master key) is known only to the gateway. In production this maps to network policies or mTLS between the gateway and upstreams. The demo shows that a direct request from the agent to Ollama does not get through. An employee connecting straight to a public provider API is a matter of company egress policy, outside this project's scope.

## Identity and permission model

Every request carries two identities: the person who triggered the agent and the agent itself. The decision is made on the intersection of their rights, and session risk can only narrow it, never widen it.

**Identity.** The agent sends `Authorization: Bearer <JWT>` with claims `sub` (user), `act.sub` (agent), `roles`, `session_id`, `exp`. The claim shape follows OAuth Token Exchange (RFC 8693). For the demo, tokens are issued by `/auth/demo-token`. An agent without a user token gets only its own rights, flagged `autonomous: true`.

**Credentials.** Real keys to databases, APIs and models live only in the gateway. The agent knows only the gateway URL, so it cannot bypass the layer for managed resources.

**Effective permissions** are computed per call:

```latex
E = U(\text{user}) \cap A(\text{agent}) \cap T(\text{task}) \setminus R(\text{risk}, \text{taint})
```

- `U`: the user's role permissions, e.g. `read:db:sales.*`.
- `A`: the agent's maximum scope, e.g. DataBot may only `read` and `generate`.
- `T`: optional task scope passed in the token (`scope`), unrestricted by default.
- `R`: actions removed by risk rules.

**Risk score and taint.** A session starts at `risk = 0`. Each control returns a `risk_delta`, and the score decays over time. Taint is set when untrusted content reaches the agent's context: a tool result flagged `untrusted` (web, email, external document) or a prompt-injection detector hit. Taint lasts until the session ends.

**The reaction depends on session type.** The token decides the type: a human `sub` means an interactive session; `autonomous: true` on an agent registered in the policy means an autonomous system.

| Session state | Interactive session (human) | Autonomous system (registered) |
| --- | --- | --- |
| `taint = true` | `write`, `delete`, `egress` removed until session end | `write`, `delete`, `egress` need human approval |
| `risk > 0.5` | `write` needs approval, denied actions get a 5 min cooldown | Throttling with backoff + alert |
| `risk > 0.8` | Tools frozen for 5 min, `generate` only | Everything except `read` goes to the approval queue + alert |

Automation never permanently revokes rights granted in the policy. It can slow down, hold for approval, or freeze a human's session for a few minutes. Permanently disabling an agent (kill switch, `POST /admin/kill`) is always a human decision.

**Human in the loop.** On `require_approval` the MCP proxy returns a tool error with an `approval_id`. Approvers see the queue at `GET /admin/approvals` (plus a CLI and a Grafana panel). After approval the agent retries the call with the same `approval_id`. No decision within the timeout means deny.

Row-level filtering is done by the adapter, not the model. For Postgres the gateway sets `app.user_id` via `set_config` and relies on RLS, so `COUNT(*)` returns the number of rows visible to that user.

## Pipeline and interfaces

The core knows nothing about SQL or any specific provider. Every interaction is normalized into one envelope: a new tool means a new adapter, and a new guardrail means a new `Control` class plus a YAML entry.

**Order for every call:**

1. Authentication: verify the JWT, load session state.
2. Adapter: normalize into `Interaction` (action, resource).
3. `pre` controls: deterministic first (cheap), semantic only when needed.
4. Policy decision: effective permissions + control verdicts → allow / redact / block / require\_approval.
5. Execute on the upstream (model, MCP server).
6. `post` controls: redact the result, set taint, check the model that actually answered.
7. Audit and metrics, update session state.

```python
@dataclass
class Interaction:
    session_id: str
    principal: str          # user (sub)
    actor: str              # agent (act.sub)
    channel: Literal["llm", "mcp", "a2a"]
    action: Literal["read", "write", "delete", "execute", "egress", "generate"]
    resource: str           # "db:sales.orders", "http:api.stripe.com", "model:qwen3:8b"
    payload: Any
    context: SessionContext # risk, taint, budgets, call history

@dataclass
class Verdict:
    decision: Literal["allow", "redact", "block", "require_approval"]
    control_id: str
    reason: str
    risk_delta: float = 0.0
    redactions: list[Span] = field(default_factory=list)
    latency_ms: float = 0.0

class Adapter(Protocol):
    def matches(self, raw: RawCall) -> bool: ...
    def normalize(self, raw: RawCall, ctx: SessionContext) -> list[Interaction]: ...

class Control(Protocol):
    id: str
    stage: Literal["pre", "post"]
    kind: Literal["deterministic", "semantic"]
    async def evaluate(self, i: Interaction, cfg: dict) -> Verdict: ...
```

One request can produce several envelopes: an SQL query reading two tables becomes two `Interaction`s, each checked separately. The strictest verdict wins (block > require\_approval > redact > allow).

`GenericMCPAdapter` handles any MCP server out of the box by mapping tool annotations (`readOnlyHint`, `destructiveHint`, `openWorldHint`) to `action`. A tool without annotations is treated as `execute` and needs an explicit rule.

## Control catalog

Deterministic controls always run and are cheap (target p95 < 5 ms combined). Semantic controls run in tiers: a small classifier on every input, and an LLM judge only when the classifier score falls in the uncertain band.

| ID | Type | Stage | Channel | What it does | Modes |
| --- | --- | --- | --- | --- | --- |
| `authn` | det. | pre | all | Verify JWT, agent and role exist | block |
| `authz` | det. | pre | all | Effective permissions vs `action:resource` | block, require\_approval |
| `model_allowlist` | det. | pre+post | llm | Allowed models per role/agent; also checks the model in the upstream response | block, log\_only |
| `pii` | det. | pre+post | all | Presidio + Polish regexes (PESEL, NIP, IBAN) | redact, block, log\_only |
| `secrets` | det. | pre+post | all | API keys, tokens, private keys, connection strings | redact, block |
| `sql_guard` | det. | pre | mcp (SQL) | Parse, classify read/write/DDL, `EXPLAIN` + cost threshold, forced `LIMIT` and timeout | block |
| `egress` | det. | pre | mcp (HTTP) | Domain allowlist, block under taint | block, require\_approval |
| `signatures` | det. | pre+post | all | Match against an external attack signature feed | block, log\_only |
| `tool_pinning` | det. | pre | mcp | Hash MCP tool descriptions, detect changes after approval (rug pull) | block |
| `budget` | det. | pre+post | all | Tokens, cost, tool calls, GPU time per user/agent/session | block |
| `loop_detect` | det. | pre | all | Same call N times within window T | block |
| `prompt_injection` | sem. | pre+post | llm, mcp | Classifier on prompts and on tool results (indirect injection) | block, log\_only |
| `tool_poisoning` | sem. | pre | mcp | Classifier on MCP tool descriptions at registration | block |
| `intent_judge` | sem. | post | llm | LLM judge compares a `tool_call` with the user's goal | require\_approval, log\_only |
| `output_policy` | sem. | post | llm | Does the answer reveal data outside the session scope | redact, block |

**Historical attacks.** The `signatures` control reads a feed from a URL or file (JSON, refreshed every N seconds) in the form `{id, source, pattern_type, pattern, severity, channels}`. Pattern types: regex, pickle opcode sequences in model files, MCP tool names and descriptions, path globs. The starter feed is built from public sources (OWASP Top 10 for LLM, documented MCP tool poisoning incidents, known model deserialization attacks). The feed is separate from the code, so a judge can add a signature and see the effect without a restart.

**Budgets.** Cost uses `litellm.cost_per_token` for commercial models and a configurable price per 1k tokens or per GPU second for local ones. Counters are atomic in Redis with periodic resets (day, session). Crossing the soft limit raises a warning in Grafana; crossing the hard limit blocks.

**Operator levers.** Beyond control verdicts, the operator has three levers. First, blocking a user, agent or use case via `blocklist` in the YAML. Second, throttling with exponential backoff, like failed logins, applied automatically to autonomous systems at elevated risk. Third, a manual per-agent kill switch via `POST /admin/kill`.

## Policy file

One source of truth: `policy.yaml`, validated with a Pydantic schema and reloaded without restart. An invalid file never replaces a working one: the gateway logs the error and keeps the last valid version.

```yaml
version: 3
profile: strict            # strict | balanced | permissive
default: deny              # allowlist: anything not in the policy is forbidden

upstreams:                 # internal network, reachable only from the gateway
  llm: { base_url: http://ollama:11434/v1 }   # or LiteLLM
  mcp:
    sales_db: { url: http://mcp-postgres:8000/mcp, adapter: sql, trust: internal }
    web:      { url: http://mcp-fetch:8000/mcp,    adapter: http, trust: untrusted }

roles:
  analyst: { allow: ["read:db:sales.*", "generate:model:*"] }
  intern:  { allow: ["read:db:sales.orders", "generate:model:qwen3:8b"] }
  admin:   { allow: ["*:*"] }

agents:
  databot:     { type: interactive, max_actions: [read, generate], deny: ["egress:*"] }
  nightly_etl: { type: autonomous,  max_actions: [read, write], approvers: [ops-team] }

risk_rules:
  interactive:
    - when: { taint: true }   then: { deny_actions: [write, delete, egress] }
    - when: { risk_gt: 0.5 }  then: { actions: [write], mode: require_approval, cooldown_s: 300 }
    - when: { risk_gt: 0.8 }  then: { freeze_tools: true, duration_s: 300 }
  autonomous:
    - when: { taint: true }   then: { actions: [write, delete, egress], mode: require_approval }
    - when: { risk_gt: 0.5 }  then: { throttle: { max_actions: 1, per_s: 10 }, alert: true }
    - when: { risk_gt: 0.8 }  then: { actions: [write, delete, execute, egress], mode: require_approval, alert: true }

approvals: { timeout_s: 600, on_timeout: deny }

blocklist:                 # manual blocks, effective right after reload
  users: []
  agents: []
  use_cases: []            # e.g. "egress:http:pastebin.com"

controls:
  pii:              { mode: redact, threshold: 0.6, entities: [PESEL, EMAIL, IBAN, PHONE] }
  secrets:          { mode: block }
  prompt_injection: { mode: block, threshold: 0.85, judge_band: [0.5, 0.85] }
  sql_guard:        { mode: block, max_cost: 10000, force_limit: 500, timeout_ms: 3000 }
  signatures:       { mode: block, feed: http://feed:9000/signatures.json, refresh_s: 30 }
  loop_detect:      { mode: block, max_repeats: 5, window_s: 60 }
  throttle:         { backoff: exponential, base_s: 5, max_s: 300 }

budgets:
  per_user:    { daily_tokens: 200000, daily_cost_usd: 2.0 }
  per_agent:   { daily_tokens: 1000000 }
  per_session: { tool_calls: 50, gpu_seconds: 120 }
  soft_limit_pct: 80
```

**Strictness profiles** change the default mode of every control with one field; a setting on a specific control always wins:

| Profile | Control with a detection | No rule for a resource | Taint in an interactive session |
| --- | --- | --- | --- |
| strict | block | deny | removes write/delete/egress |
| balanced | redact or require\_approval | deny | removes egress |
| permissive | log\_only | deny | log only |

Default deny applies in every profile. A profile changes how detections are handled, but never opens resources outside the allowlist.

**Hot reload:** a file watcher (`watchfiles`) and `POST /admin/reload`. Every change gets a version hash that goes into every audit entry and into a Grafana annotation, so you can see exactly when the new policy took effect.

## Audit, metrics and Grafana

Every decision leaves one audit entry with the full "why", and metrics feed dashboards for both security and management.

**Audit entry** (JSON to stdout → Loki, plus JSONL to an export file):

```json
{"ts":"2026-10-04T10:12:03Z","session_id":"s-81f","principal":"intern@demo","actor":"databot",
 "channel":"mcp","action":"read","resource":"db:sales.customers","decision":"block",
 "verdicts":[{"control":"authz","decision":"block","reason":"resource outside effective scope"}],
 "effective_scope":["read:db:sales.orders"],"risk":0.35,"taint":false,
 "policy_version":"a1c9e2","latency_ms":{"total":7.4,"controls":{"authz":0.2,"pii":3.1}}}
```

The full payload is never logged: we store a hash and redacted excerpts, so the audit log does not become a leak itself.

**Prometheus metrics** (`/metrics`):

- `acl_requests_total{channel,decision,agent}`
- `acl_control_verdicts_total{control,decision}`
- `acl_control_latency_seconds{control}` (histogram) and `acl_overhead_seconds{channel}` (gateway overhead excluding upstream)
- `acl_tokens_total{user,agent,model}`, `acl_cost_usd_total{user,agent,model}`
- `acl_budget_usage_ratio{scope,id}`
- `acl_session_risk{session}`, `acl_tainted_sessions`
- `acl_approvals_pending`, `acl_throttled_total{agent}`
- `acl_policy_reloads_total{result}`

| Dashboard | Audience | Panels |
| --- | --- | --- |
| Posture | management | Block rate, top threats, cost per team/agent, budget usage, daily trend |
| Threats | security | Stream of blocks from Loki, top controls, highest-risk sessions, signature hits, approval queue, policy change annotations |
| Session trace | security | Timeline of one session: decisions, effective permissions, risk score (`session_id` variable) |
| Performance | judges, ops | p50/p95/p99 overhead per channel and per control, throughput |

Dashboards and data sources are provisioned as files in the repo, so `docker compose up` brings up a ready Grafana.

## Test suite and demo

The competition rules weight the test suite at 20%, so every control has at least one allow/deny pair and the whole suite runs with one command: `make test`.

**Test layers:**

- `tests/unit/`: each control on its own, with a case table (`pytest.mark.parametrize`): input → expected verdict.
- `tests/policy/`: role × agent × action × resource matrix → allow/deny. Same request, different user, different result. Resources and tools outside the policy are denied in every profile (default deny).
- `tests/session_modes/`: the same event (taint, high risk) removes actions in an interactive session, while an autonomous session gets throttling and the approval queue, without losing rights.
- `tests/approvals/`: `require_approval` → queue → approve, deny, timeout; manual kill switch and blocklist take effect after reload.
- `tests/bypass/`: a direct request from the agent network to Ollama, the router or MCP servers does not get through.
- `tests/e2e/`: real requests through the gateway to Ollama and MCP servers in docker compose.
- `tests/attacks/`: a corpus of attack prompts and payloads (direct and indirect injection, secret leakage, tool poisoning, pickle in a model file) with the expected block.
- `tests/budget/`: token and call budget exhaustion, loop detection, throttling with backoff.
- `tests/reload/`: changing `policy.yaml` mid-test changes the verdict without restart; a broken file does not break the working policy.
- `tests/perf/`: p50/p95 overhead with a mocked upstream, written into the report.

Output: JUnit XML + HTML report (`pytest-html`) summarizing positive and negative cases per control.

**Demo script (3 minutes):**

1. Anna (analyst) and Bartek (intern) send DataBot the same prompt asking for the customer count. Anna gets the full result; Bartek gets only his rows or a block.
2. DataBot reads a page with hidden injection through the `web` tool. The session is tainted, the attempt to write a report is blocked, and the risk score rises in Grafana.
3. The same injection hits the autonomous `nightly_etl`. The write is not blocked; it waits in the approval queue, and the process slows down instead of stopping.
4. The agent generates a heavy query. `sql_guard` rejects it after `EXPLAIN`.
5. A prompt with an API key and a PESEL number: the secret is blocked, the PESEL is redacted.
6. The agent tries to bypass the gateway and call Ollama directly. The connection does not go through.
7. A judge raises `max_cost` in `policy.yaml` or adds a signature to the feed. The same request gets a different verdict, and Grafana shows an annotation for the new policy version.

## Stack, repo layout and schedule

Python throughout, everything local in `docker compose`, no paid APIs. We check the license of every dependency before using it.

| Layer | Choice | Notes |
| --- | --- | --- |
| Proxy | FastAPI + httpx, uvicorn | SSE streaming for LLM and MCP |
| LLM client / cost | `litellm` as a library | We do not use LiteLLM Proxy |
| MCP | official MCP Python SDK | Proxy acts as an MCP server to the agent and a client to upstreams |
| Policies | Pydantic + our own evaluator | Option: Cedar (`cedarpy`) if time allows |
| PII | Presidio + Polish regexes |  |
| SQL | `sqlglot`, Postgres with RLS |  |
| Injection classifier | small Hugging Face model on CPU | Check the license before choosing |
| LLM judge and demo models | Ollama (e.g. Qwen3 8B) |  |
| Session state, budgets, approvals | Redis |  |
| Telemetry | prometheus-client, Loki, Grafana | Provisioned from the repo |
| Demo agent | simple Python agent with MCP | Not assessed |

```
ai-control-layer/
  gateway/
    main.py            # FastAPI, two routers: /v1 (LLM), /mcp/{server}, plus /admin
    pipeline.py        # step order, verdict merging
    identity.py        # JWT, delegation, session state, session type
    policy/            # schema, loader, hot reload, evaluator
    adapters/          # generic_mcp, sql, http, fs, llm
    controls/          # one class = one file
    approvals.py       # approval queue, throttling, kill switch
    telemetry.py       # metrics, audit
  feeds/signatures.json
  policy.yaml
  demo/                # agent, database seed, MCP servers
  grafana/             # dashboards/, provisioning/
  tests/
  docker-compose.yml   # upstreams on an internal network only
  Makefile             # up, test, demo, report
```

**Schedule (hours from now, 22h):**

1. 0–5: gateway skeleton, LLM proxy to Ollama, generic MCP proxy, JWT and session state, policy loader with default deny and hot reload, upstream isolation in the compose network.
2. 5–10: deterministic controls (authz, PII, secrets, budgets, loops, signatures, model allowlist), SQL adapter with RLS and `EXPLAIN`.
3. 10–14: semantic controls, taint and risk rules per session type, approval queue, throttling, blocklist, kill switch, tool pinning.
4. 14–18: test suite, then Grafana dashboards and Loki.
5. 18–22: diagram, demo recording, 10-slide PDF, buffer.

Priority, per the challenge owner's advice: security first. Prometheus metrics are exposed from the first stage because they are cheap, but dashboards come only once controls work and are tested.

Team split: core and pipeline, adapters, controls, telemetry and Grafana, tests and demo. Agree on the `Adapter` and `Control` interfaces in the first hour; after that everyone works independently.

## Competitive landscape and differentiators

The market is crowded and every individual piece of this project already exists somewhere. The differentiator is not the parts but how they are coupled: threat detections directly reshape delegated permissions, and enforcement goes below the tool level, down to the data.

| Solution | What it does | What it does not do (or not in this combination) |
| --- | --- | --- |
| [Invariant Gateway + Guardrails](https://invariantlabs.ai/blog/guardrails) (now under Snyk as [agent-scan](https://deepwiki.com/snyk/agent-scan)) | Transparent LLM and MCP proxy plugged in via base URL, data-flow rules toward untrusted sinks, PII, secrets, toxic flow analysis | No user ∩ agent identity model; rules govern data flows, not a user's rights to data |
| [Monosign MCP Gateway](https://monofor.com/sign/mcp-gateway) | Agent and human identity, on-behalf-of as agent ∩ user, per-user budgets, approval for destructive tools, credentials in a vault | MCP only, tool-level authorization; no LLM path and no semantic detectors coupled to permissions |
| [TrueFoundry MCP Gateway](https://www.truefoundry.com/docs/agent-platform/agent-governance/truefoundry-implementation/microsoft-foundry-agents) | Agent registry, on-behalf-of with user and agent identity, token exchange to the MCP server | Commercial platform; no taint or risk-based narrowing of rights |
| [Traefik MCP Gateway](https://doc.traefik.io/traefik-hub/mcp-gateway/guides/mcp-gateway-best-practices) | Task-based access control (TBAC) on JWT claims, on-behalf-of by forwarding the header | Part of Traefik Hub; semantic guardrails out of scope |
| [Kong AI Gateway](https://dev.to/rebeccaws/15-best-mcp-gateways-for-developers-in-2026-4j9j), Portkey, Bifrost | LLM routing, budgets, caching, prompt guards, MCP proxy in one layer | Routers with security add-ons; [Kong's MCP gateway is Enterprise-only](https://airbyte.com/agentic-data/ai-agent-api-gateway) |
| [tracewall](https://pypi.org/project/tracewall/), [ShisaD](https://pypi.org/project/shisad/0.5.1/) | Per-session taint and lethal trifecta blocking, policy pipeline on every tool call | Local tools for a single agent; no organizational identity, roles or company reporting |
| [Datadog AI Guard](https://docs.datadoghq.com/security/ai_guard) | LLM evaluator on every prompt and tool call with full session context | Commercial service; allow/deny without a permission model |

**What we solve that they do not combine:**

1. **Detection → authorization.** Elsewhere a detector blocks a request. Here a hit raises session risk and changes effective permissions: a human's session loses specific actions, while an autonomous system gets throttling and human approval. The change is visible in the audit log as a change in effective permissions.
2. **Authorization at the data level, not the tool level.** MCP gateways answer "may the agent call `query`?". We answer the question the challenge owner raised: which rows does this `COUNT` see for this user, and how expensive may this query be.
3. **One policy for LLM and tools, with shared session state.** Intent from the model's response and enforcement in MCP are correlated via `session_id`.
4. **A security layer in front of any router.** We do not replace LiteLLM, Kong or Portkey; we sit in front of them.
5. **Open, local, editable live.** One YAML file, an external signature feed, Grafana from the repo, zero paid dependencies.

The honest pitch line: we do not claim to have invented taint or on-behalf-of. We show that only their combination in one layer closes the agent permission problem the challenge owner called the biggest one.

**Sources** (as of 2026-10-03): links in the table.
