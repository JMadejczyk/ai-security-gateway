# AI Control Layer — tech spec

Oct 3, 2026 · @Jakub · v2 (after review, see [Changelog](#changelog))

## Goal and thesis

We are building a security layer that all agent traffic passes through, both to models and to tools. It is not an LLM router. It sits in front of any OpenAI-compatible gateway (LiteLLM, Ollama, OpenRouter) and in front of MCP servers.

Thesis: detecting a threat should not just block one request. It should change what the agent may do for the rest of the session. In an interactive session that means narrowing permissions. For a registered autonomous system it means throttling and human approval instead: configured grants are preserved while execution is temporarily withheld. Base permissions are always principal permissions ∩ agent permissions ∩ task scope; session risk can only add restrictions and obligations on top, never widen them.

The problem raised by the challenge owner: an agent acts on behalf of a specific person but must not inherit all of that person's rights, and the same agent triggered by two people must see different data.

## Scope

In 22 hours we ship two entry points (LLM and MCP), a shared policy core with hot reload, a dozen-plus controls, Grafana and a test suite. Everything else is an extension point shown on the diagram.

| Area | In scope | Out of scope (extension point) |
| --- | --- | --- |
| Entry points | LLM proxy (OpenAI-compatible `/v1/chat/completions`, `/v1/models`), MCP proxy (streamable HTTP, tools-only subset) | A2A proxy, in-process SDK/library, MCP resources/prompts/sampling/elicitation |
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

**No bypassing the gateway.** Upstreams accept traffic only from the gateway. Containers on a shared Docker bridge can reach each other, so "internal network, no published ports" alone is not enough. The topology is:

- `edge` network (`internal: true`): the agent and the gateway's agent listener (`:8080`, serving `/v1` and `/mcp`). The gateway is the only service the agent can resolve or reach, and the agent has no internet access.
- `ops` network: the gateway's operator listener (`:9090`, serving `/auth/demo-token`, `/admin/*`, `/metrics`), Prometheus, Loki, Grafana Alloy and Grafana. Only this listener and Grafana are published, on `127.0.0.1`.
- `llm_backend` network (`internal: true`): the gateway and Ollama (and LiteLLM if used).
- `mcp_backend` network (`internal: true`): the gateway and the internal MCP servers (`mcp-postgres`, `mcp-files`) plus Postgres.
- `mcp_untrusted` network (`internal: true`): the gateway and `mcp-fetch`. A compromised fetch server cannot reach Postgres or Ollama.
- `state` network (`internal: true`): the gateway and Redis (`redis:7.2`, BSD-3). Session state, budgets, loop-detection counters, approvals and the kill switch live here.
- `fetch_egress` network: `mcp-fetch` only, the one runtime network with outbound internet. (Kept separate from `mcp_untrusted` because Docker routes a container's published ports via its default-route network; a non-internal network shared with the gateway would capture the gateway's host port mapping.)
- `bootstrap` network (external): only the one-shot `ollama-init` and `models-init` jobs, which pull models into volumes before lock-down.
- `demo_web` network (`internal: true`, demo overlay `demo/compose.demo.yml` only): `mcp-fetch` and a static `demo-web` page for demo scene 2. The overlay sets two exemptions from the private-address rule, each bound to one name, one address and one origin (`demo-web=10.218.97.10`, `http://demo-web:80`): `ACL_EGRESS_DEMO_HOSTS` (gateway, audited as `egress_demo_host`) and `ACL_FETCH_DEMO_HOSTS` (mcp-fetch). Any other resolved address, scheme or port is refused; both are empty by default, and tests pin that.
- No upstream publishes a port to the host. No container gets the Docker socket, `privileged` or host networking.

The router key (e.g. the LiteLLM master key) is known only to the gateway; gateway bearer tokens are never forwarded upstream. Ollama models are pulled before the network is locked down (an init step on a separate, temporary network). In production this maps to network policies or mTLS between the gateway and upstreams. The threat boundary excludes a hostile host or Docker administrator. The bypass tests run **from inside the agent container** against Ollama, MCP servers and Postgres. An employee connecting straight to a public provider API is a matter of company egress policy, outside this project's scope.

## Identity and permission model

Every request carries two identities: the person who triggered the agent and the agent itself. The decision is made on the intersection of their rights, and session risk can only narrow it, never widen it.

**Identity.** The agent sends `Authorization: Bearer <JWT>` with claims `iss`, `aud`, `sub` (principal), `act.sub` (agent), `roles`, `mode` (`interactive` | `autonomous`), `session_id`, `sid_iat` (when the session id was minted; a token is refused once `sessions.max_lifetime_s` has passed since it), optional `scope`, `iat`, `exp`. Session ids are always minted by the issuer, never chosen by the caller, and tombstones outlive every token that could name them, so an ended session cannot be revived without the signing key. The claim shape follows OAuth Token Exchange (RFC 8693); the shape alone does not prove delegation, so the gateway also checks it:

- Algorithm pinned (`HS256` for the demo key; `RS256`/`EdDSA` in production), signature, `iss`, `aud = ai-control-layer`, `exp`, and a maximum token lifetime (default 1 h).
- `act.sub` must be an agent registered in the policy; `mode` must equal that agent's registered `type`; the principal must be allowed for the agent (`principals` on the agent, default: any human principal for interactive agents).
- `roles` must exist in the policy.
- A session is bound on first use to `(sub, act.sub, mode)`. A token presenting the same `session_id` with a different binding is rejected. Refreshing a token never clears session state (taint, risk).

For an **autonomous** agent there is no human. `sub` is the agent's own service principal (`svc:<agent>`), and `U` is the agent's own `allow` grants, so the formula below is unchanged. A missing user never silently selects autonomous mode: `mode` is asserted by the issuer and validated against the registration.

Demo identities: `anna@demo` (analyst), `bartek@demo` (intern), `olga@demo` (ops-team), `root@demo` (admin), and the service principal `svc:nightly_etl`. Tokens come from `POST /auth/demo-token`, which only issues tokens for identities predefined in `demo/identities.yaml` and is bound to the operator network (disabled when `ACL_DEMO_TOKENS=0`). The signing key lives only in the gateway, never in the agent container. `/admin/*` requires an **operator token**: a separate audience (`ai-control-layer-operator`), no `act` claim and no session, issued only to identities with `admin` or an approver role. Role `admin` may do everything; an approver role may only list and decide approvals for the agents that name it in `approvers`. The agent API refuses operator tokens and the operator API refuses agent tokens. Bearer-token theft still means impersonation; we do not claim workload attestation.

**Credentials.** Real keys to databases, APIs and models live only in the gateway. The agent knows only the gateway URL, so it cannot bypass the layer for managed resources.

**Decision model.** Every call is decided in two layers.

```text
base_allowed =
      principal grants match (action, resource)      # U: union of the principal's role grants
  AND agent grants match (action, resource)           # A: agent's allow list, filtered by max_actions
  AND task scope matches (action, resource)           # T: token `scope`; absent = unrestricted, [] = nothing
  AND no explicit deny matches                        # agent deny, blocklist

if not base_allowed or agent killed or principal/agent/use case blocklisted:
    deny
else:
    apply session restrictions (risk rules for the session mode)    # can only remove or condition
    run controls, collect verdicts and obligations (approval, throttle, redaction)
    execute only after every obligation is satisfied
```

Grants are matched against the concrete `(action, resource)` of each interaction; we never intersect pattern sets symbolically (that would wrongly decide `read:db:sales.*` and `read:db:sales.orders` do not overlap). Approval and throttling are obligations on an already allowed call; **an approval never grants an operation outside `base_allowed`**.

**Permission grammar.**

```text
permission       := action ":" resource_pattern        # split at the FIRST colon only
action           := "read" | "write" | "delete" | "execute" | "egress" | "generate" | "*"
resource_pattern := "*" | namespace ":" identifier_pattern   # split at the next colon, later colons preserved
```

`generate:model:qwen3:8b` → action `generate`, namespace `model`, identifier `qwen3:8b`. `*` matches any run of characters (it crosses `.` and `:`), there are no other wildcards, matching is case-sensitive and anchored on the whole string, `*` as the whole resource pattern matches every resource, so `*:*` is every action on every resource and `egress:*` is egress anywhere. A concrete resource (on an `Interaction`) always has the `namespace:identifier` form.

**Risk score and taint.** A session starts at `risk = 0`. Risk is bounded to `[0, 1]`. Each control verdict adds its `risk_delta` (clamped), including verdicts on calls that end up blocked. Risk decays exponentially with a configurable half-life (`risk.half_life_s`, default 600 s, wall clock on the gateway). Thresholds are strict (`risk_gt: 0.5` means `> 0.5`). Each detection control has a configurable `risk_delta` (defaults: `prompt_injection` 0.6, `secrets` 0.3, `signatures` 0.4, `pii` 0.1, `egress` 0.3, `authz` deny 0.1); an allow verdict adds nothing. Taint is set when untrusted content reaches the agent's context: a tool result from a server with `trust: untrusted` (web, email, external document) or a prompt-injection detector hit. A tool result that is blocked or replaced still taints the session, because the attempt reached the gateway on the agent's behalf. **Taint lasts until the session ends, and decay never clears it.** Taint from a tool result is persisted before that result is released to the agent. Calls within one session are serialized, so a call never races past a restriction set by the previous one. A session ends on `DELETE /v1/session`, after `sessions.idle_ttl_s` (default 3600) without calls, or after `sessions.max_lifetime_s` (default 86400); a token naming an ended session is rejected and a new `session_id` must be issued. `/v1/models` lists only the models the session may `generate`.

**The reaction depends on session mode.** The mode is the token's `mode` claim, validated against the agent's registered `type`.

| Session state | Interactive session (human) | Autonomous system (registered) |
| --- | --- | --- |
| `taint = true` | `write`, `delete`, `egress` removed until session end | `write`, `delete`, `egress` need human approval |
| `risk > 0.5` | `write` needs approval; a call denied in this state starts a 5 min cooldown on the same `action:resource` | Throttling with exponential backoff + alert |
| `risk > 0.8` | Tool calls (MCP channel) frozen for 5 min, LLM `generate` still allowed | Every action except `read` and `generate` goes to the approval queue + alert |

A `cooldown_s` starts when a call is denied in that state and applies to the same `action:resource`; a `freeze_tools` `duration_s` starts when the threshold is first crossed. Neither timer is refreshed by further calls; when the timer ends the rule applies again only if risk is still above the threshold, so a freeze can be re-entered but never extends itself indefinitely. Automation never permanently revokes rights granted in the policy. It can slow down, hold for approval, or freeze a human's session for a few minutes. `throttle` caps an autonomous agent at `max_actions` per `per_s`; each call over the cap is rejected with `Retry-After`, and consecutive rejections grow it exponentially from `throttle.base_s` up to `throttle.max_s`, resetting after one compliant window. Approval timeout means deny, so an autonomous process may still stop; availability never overrides authorization. Permanently disabling an agent (kill switch, `POST /admin/kill` / `POST /admin/unkill`, persisted in Redis, checked at every pipeline step) is always a human decision.

**Human in the loop.** On `require_approval` the MCP proxy returns a tool error with an `approval_id`. An approval authorizes **one exact pending operation**: it is bound to principal, agent, session, server, tool, a digest of the canonical (sorted-key JSON) arguments and the policy version, and expires with `approvals.timeout_s`. Approvals live in Redis next to budgets. Approvers see the queue at `GET /admin/approvals` and decide with `POST /admin/approvals/{id}/approve|deny` (plus `python -m gateway.cli approvals …` and a Grafana panel); an approver must hold the agent's `approvers` role and can never be the session's own principal. A pending approval expires `timeout_s` after it is requested; once approved, it is usable for `timeout_s` after the decision. The binding also covers the full `action:resource` set and the digest of the final (redacted, rewritten) operation, so a reload that changes either needs a new approval. A kill raises a per-agent generation that atomically voids every open approval of that agent. On retry with the `approval_id` the gateway re-authenticates, verifies the binding, re-checks base authorization, blocklist, kill switch and controls, satisfies only the approval obligation, and atomically moves the operation `approved → executing`. States: `pending`, `approved`, `denied`, `expired`, `executing`, `succeeded`, `failed`, `uncertain` (upstream outcome unknown after a lost response; never auto-retried). A retry of a pending operation returns the same `approval_id` instead of creating a new entry, and approval retries are excluded from loop detection. Exactly-once side effects are not guaranteed for upstreams without idempotency keys.

**Row-level filtering** is done by the data layer, not the model. The SQL MCP server is trusted (`trust: internal`) and owns the database transaction. The gateway forwards the authenticated principal in a header signed with an internal key (`X-ACL-Principal`, never the agent's bearer token). For each call the server takes one pooled connection, begins a transaction, runs `SELECT acl.set_principal($1)` (a `SECURITY DEFINER` wrapper over transaction-local `set_config`, usable once per transaction; `EXECUTE` on `set_config` is revoked from the application role, so no statement can change the principal or its timeouts), executes the exact query `sql_guard` approved, then commits or rolls back before returning the connection. The `X-ACL-Principal` assertion also carries the execution limits from the policy (`stmt_timeout_ms`, `max_rows`, `max_result_bytes`), so the server applies `SET LOCAL statement_timeout`/`lock_timeout` and result caps it cannot be talked out of by the agent. `sql_guard` is a *sealing* control: it runs after every other pre control's rewrites and redactions, prices and canonicalizes the final SQL, and the pipeline refuses to dispatch anything that differs from what it sealed (`sql_changed_after_guard`). It gets its plan cost from an internal `explain` tool on the same server: same principal, same transaction setup, `EXPLAIN (FORMAT JSON)` without `ANALYZE`. That tool is called only by the gateway; it is absent from the operator mapping, so agents can never call it. A missing principal fails closed. The database role is a non-owner without superuser or `BYPASSRLS`, tables use `FORCE ROW LEVEL SECURITY`, and `sql_guard` rejects any statement that could touch `set_config`/`current_setting` or session state. `COUNT(*)` therefore returns the number of rows visible to that user.

## Pipeline and interfaces

The core knows nothing about SQL or any specific provider. Every interaction is normalized into one envelope: a new tool means a new adapter, and a new guardrail means a new `Control` class plus a YAML entry.

**Order for every call:**

1. Authentication: verify the JWT, load and lock session state (one call per session at a time).
2. Adapter: normalize into one or more `Interaction`s (action, resource).
3. Base authorization (decision model above) on every interaction; any deny stops here.
4. `pre` controls: deterministic first (cheap), semantic only when needed.
5. Merge verdicts → allow / redact / block / require\_approval, plus obligations.
6. Execute on the upstream (model, MCP server) **once** for the original request, only if every interaction passed.
7. `post` controls on the complete (buffered) result: redact, set taint, check the model that actually answered.
8. Persist session state (risk deltas from every verdict, including blocked calls), then audit and metrics, all before releasing the result.

Every decision reads one immutable policy snapshot taken at step 1, so a reload mid-call never mixes two policy versions.

```python
Action = Literal["read", "write", "delete", "execute", "egress", "generate"]
Decision = Literal["allow", "redact", "block", "require_approval"]
Stage = Literal["pre", "post"]

class Interaction(BaseModel, frozen=True):
    session_id: str
    principal: str            # sub (human, or svc:<agent> for autonomous)
    actor: str                # act.sub
    mode: Literal["interactive", "autonomous"]
    channel: Literal["llm", "mcp", "a2a"]
    server: str | None = None # MCP server name from the policy, None for llm
    action: Action
    resource: str             # "db:sales.orders", "http:api.stripe.com", "model:qwen3:8b"
    payload: Any              # request payload (pre) ...
    result: Any = None        # ... and upstream result (post)
    context: SessionContext   # risk, taint, budgets, call history (read-only snapshot)

class Span(BaseModel, frozen=True):
    path: str                 # JSON pointer into payload/result, e.g. "/messages/0/content"
    start: int                # code-point offsets within that string
    end: int
    label: str                # "PESEL", "API_KEY", ...

class Verdict(BaseModel, frozen=True):
    decision: Decision
    control_id: str
    reason_code: str          # structured, e.g. "resource_outside_scope"
    reason: str = ""          # human text, never contains payload data
    enforced: bool = True     # False = log_only: recorded, decision not applied
    risk_delta: float = 0.0
    redactions: tuple[Span, ...] = ()
    rewrite: Any = None       # replacement payload (pre) or result (post), e.g. sql_guard's LIMIT-ed query
    latency_ms: float = 0.0

class Adapter(ABC):
    @abstractmethod
    def matches(self, raw: RawCall) -> bool: ...
    @abstractmethod
    def normalize(self, raw: RawCall, ctx: SessionContext) -> list[Interaction]: ...

class Control(ABC):
    id: ClassVar[str]
    stages: ClassVar[frozenset[Stage]]          # a control can run pre, post, or both
    kind: ClassVar[Literal["deterministic", "semantic"]]
    mandatory: ClassVar[bool] = False           # enforces in every profile, cannot be set to log_only
    @abstractmethod
    async def evaluate(self, i: Interaction, stage: Stage, cfg: ControlConfig) -> Verdict: ...
```

One request can produce several envelopes: an SQL query reading two tables becomes two `Interaction`s, each checked separately; the request executes once (with any rewrites), only after all of them pass. Merging: the strictest **enforced** decision wins (block > require\_approval > redact > allow), but obligations accumulate: approval plus redaction yields both. Controls run in a fixed order; when a control returns `rewrite`, the pipeline builds a new `Interaction` (`model_copy(update=...)`) carrying it, and later controls and the upstream see the rewritten payload. The upstream executes the final payload, never the agent's original when a rewrite exists. Redaction spans are applied to the payload/result after merging; overlapping spans on the same path are unioned. `log_only` is not a decision: a control in `log_only` mode returns its would-be decision with `enforced=False`, which is audited and counted but not applied.

`GenericMCPAdapter` maps MCP tools to `(action, resource)` from an **operator-owned mapping** in the policy (`upstreams.mcp.<server>.tools`), keyed by tool name. MCP tool annotations (`readOnlyHint`, `destructiveHint`, `openWorldHint`) come from the server and are untrusted: they are shown to the operator as hints when building the mapping, but never authorize anything. A tool missing from the mapping is denied. Arguments are validated against the tool's input schema: the one pinned in `pins/<server>.json` (written by `acl pin <server>` from an operator-reviewed `tools/list` snapshot; it becomes mandatory with `tool_pinning` in stage 3), or until then the schema the upstream advertised when the gateway first connected. `tools/list` is filtered: a tool is listed only if it is mapped, its action is in the agent's `max_actions`, and the session's restrictions do not currently remove that action. Resource-level checks happen on `tools/call`, once arguments exist.

The MCP proxy supports a pinned protocol version (`2025-06-18`) and a tools-only subset: `initialize`, `notifications/initialized`, `tools/list`, `tools/call`, `ping`. Resources, prompts, sampling, elicitation and other server-initiated requests are answered with a JSON-RPC "method not found" error. Each downstream (agent) MCP session maps to its own upstream session; upstream sessions are never shared across principals.

## Control catalog

Deterministic controls always run and are cheap (target p95 < 5 ms combined, excluding Presidio's NLP pass, which is measured separately). Semantic controls run in tiers: a small classifier on every input, and an LLM judge when the classifier score falls in the uncertain band. **Exception for text the agent authored** (LLM pre stage, `user`/`system`/`developer` messages): any classifier hit at or above the band goes to the judge, because the classifier misfires on short definitional questions ("What is a primary key?" scores 1.0). The judge confirming blocks; the judge clearing allows; no answer within `user_judge_timeout_s` allows the call but taints the session and adds risk (`prompt_injection_unconfirmed`). Tool results, assistant/tool messages in the history and mixed windows keep the hard threshold and never wait for the judge. A verdict may request taint without blocking (`Verdict.taint`). Controls marked *mandatory* enforce in every profile and reject `mode: log_only`.

**Streaming.** Post controls need the complete output, and an SSE chunk that has reached the agent cannot be retracted. For `stream: true` the LLM proxy therefore calls the upstream **non-streaming**, runs every post control on the full response (text and `tool_calls`), and then re-emits the approved result to the agent as OpenAI-compatible SSE chunks ending in `data: [DONE]`. Agents that expect streaming keep working; first-token latency equals full-response latency. Responses are bounded (`limits.max_response_bytes`).

**Classifier scope.** `prompt_injection` classifies prose: text with at least 3 words and 10 letters, after HTML pages are reduced to readable blocks (scripts and styles dropped, hidden elements and comments kept) and JSON text is split into its keys and values. Numbers, dates, ids and other structured data are skipped and don't count toward `max_chars`. Gateway redaction markers are neutralised before classification. Text split across messages and tool results is re-joined in rolling windows. Measured on our 50-sample corpus at threshold 0.85: precision 0.86, recall 0.90 (Polish recall 0.75). Known false positive: the example.com boilerplate page (0.94).

**Judges.** Configured by a top-level `judges:` section (`model`, `timeout_s`, `max_content_chars`, `max_output_tokens`); configuring a judge control without it is a validation error, and with neither present the judge controls are off. `intent_judge`, `output_policy` and the `prompt_injection` judge band share one `JudgeClient` that calls the policy's LLM upstream directly (temperature 0, JSON output validated by a Pydantic model, its own timeout). A judge that times out or answers malformed JSON counts as unavailable, and the control fails closed in its enforcing mode. Judge calls are internal: they are not audited as agent requests and not charged to the agent's budget, but they are counted (`acl_judge_calls_total{control,result}`).

**Intent vs enforcement.** `session_id` correlates a conversation, not an authorized tool invocation: an agent can change arguments, call a different tool, or call MCP without any LLM request. So `intent_judge` is advisory and every MCP call goes through full authorization on its own. Its verdict never holds an LLM response: the response is released, and a `require_approval` verdict marks that `tool_call` (tool name + argument digest) as flagged in the session, so the matching MCP `tools/call` then requires approval through the normal MCP flow. It also adds its `risk_delta`. The user goal the judge compares against comes from trusted session metadata (the first user message recorded by the gateway), not from the transcript the agent sends later.

| ID | Type | Stage | Channel | What it does | Modes |
| --- | --- | --- | --- | --- | --- |
| `authn` | det. | pre | all | Verify JWT, agent and role exist, session binding (mandatory) | block |
| `authz` | det. | pre | all | Base authorization + session restrictions vs `action:resource` (mandatory) | block, require\_approval |
| `model_allowlist` | det. | pre+post | llm | Allowed models per role/agent; also checks the model in the upstream response | block, log\_only |
| `pii` | det. | pre+post | all | Presidio + Polish regexes (PESEL, NIP, IBAN) | redact, block, log\_only |
| `secrets` | det. | pre+post | all | API keys, tokens, private keys, connection strings (mandatory) | redact, block |
| `sql_guard` | det. | pre | mcp (SQL) | Parse with `sqlglot` (Postgres dialect), allow only a supported `SELECT` subset (single statement, no DDL/DML, no CTEs, schema-qualified tables only, no `COPY`, no session or settings functions, only allowlisted functions, every table resolved), `EXPLAIN (FORMAT JSON)` without `ANALYZE` against a planner-cost threshold, forced `LIMIT`, `statement_timeout`/`lock_timeout`, result-byte limit; the exact rewritten query is what executes (mandatory) | block |
| `egress` | det. | pre | mcp (HTTP) | Parsed scheme/host/port allowlist, DNS resolution checked against private, loopback and metadata ranges, redirects re-validated; taint handling comes from `risk_rules` per session mode | block, require\_approval |
| `signatures` | det. | pre+post | all | Match against an external attack signature feed (size-bounded, schema-validated, regex with a timeout, last valid feed kept) | block, log\_only |
| `tool_pinning` | det. | pre | mcp | Pin tool names, input schemas, descriptions and annotations to the approved baseline (`pins/<server>.json`, reviewed with `acl pin`); servers have `require_pin: true` by default. Any change (rug pull) quarantines the tool in Redis across sessions until an operator re-pins or clears it. Does not prove server implementation integrity | block |
| `budget` | det. | pre+post | all | Tokens, cost, tool calls, GPU time per user/agent/session | block |
| `loop_detect` | det. | pre | all | Same call N times within window T | block |
| `prompt_injection` | sem. | pre+post | llm, mcp | Classifier on prompts and on tool results (indirect injection) | block, log\_only |
| `tool_poisoning` | sem. | pre | mcp | Classifier on MCP tool descriptions at registration | block |
| `intent_judge` | sem. | post | llm | LLM judge compares a `tool_call` with the user's goal | require\_approval, log\_only |
| `output_policy` | sem. | post | llm | Does the answer reveal data outside the session scope | redact, block |

**Historical attacks.** The `signatures` control reads a feed from a URL or file (JSON, refreshed every N seconds) in the form `{version, signatures: [{id, source, pattern_type, pattern, severity, channels}]}`. Pattern types: regex, MCP tool names and descriptions, path globs (pickle opcode scanning only applies once model-file ingestion exists, so it is out of scope). The starter feed is written by us, inspired by public sources (OWASP Top 10 for LLM, documented MCP tool poisoning incidents), without copying their text. The feed is separate from the code and has its own version, recorded in audit entries next to the policy version, so a judge can add a signature and see the effect without a restart.

**Budgets.** Cost comes from a `pricing` table in the policy: per model, a price per 1k prompt and completion tokens, plus a price per GPU second for local models (`litellm.cost_per_token` is the extension point once commercial models are added; it is not a dependency now) ("GPU seconds" for Ollama are upstream wall time, an estimate, labelled as such). Budget is **reserved** atomically in Redis before dispatch (estimated from `max_tokens`, which the gateway always sets to the reserved cap; `n` other than 1 is refused). GPU time is reserved as an allowance bounded by the remaining budget and enforced as a deadline on the upstream call, so concurrent calls cannot overspend it either. Reservations and settlements carry an operation id and are idempotent; a settlement that cannot be recorded blocks its scopes until it lands and reconciled with actual usage on completion, failure or cancellation, so concurrent calls cannot overspend. Counters reset per day (UTC) and per session. Crossing the soft limit raises a warning in Grafana; crossing the hard limit blocks. If Redis is unavailable, budget-limited calls fail closed.

**Operator levers.** Beyond control verdicts, the operator has three levers. First, blocking a user, agent or use case via `blocklist` in the YAML. Second, throttling with exponential backoff, like failed logins, applied automatically to autonomous systems at elevated risk. Third, a manual per-agent kill switch via `POST /admin/kill`.

## Policy file

One source of truth: `policy.yaml`, validated with a Pydantic schema and reloaded without restart. An invalid file never replaces a working one: the gateway logs the error and keeps the last valid version.

```yaml
schema_version: 2
profile: strict            # strict | balanced | permissive
default: deny              # the only accepted value: anything not granted is forbidden

upstreams:                 # internal networks, reachable only from the gateway
  llm:
    base_url: http://ollama:11434/v1        # or LiteLLM
    api_key_env: null                       # env var holding the router key, if any
  mcp:
    sales_db:
      url: http://mcp-postgres:8000/mcp
      adapter: sql
      trust: internal
      tools:
        query: { action: read }             # sql adapter derives resources from the parsed tables
    web:
      url: http://mcp-fetch:8000/mcp
      adapter: http
      trust: untrusted
      tools:
        fetch: { action: read, resource: "web:{url}" }     # http adapter reduces {url} to its host
    reports:
      url: http://mcp-files:8000/mcp
      adapter: fs
      trust: internal
      tools:
        write_report: { action: write, resource: "fs:reports/{name}" }

roles:
  analyst:  { allow: ["read:db:sales.*", "read:web:*", "write:fs:reports/*", "generate:model:*"] }
  intern:   { allow: ["read:db:sales.orders", "read:db:sales.customers", "read:web:*",
                      "write:fs:reports/*", "generate:model:qwen3:8b"] }
  admin:    { allow: ["*:*"] }
  ops-team: { allow: [] }                   # approvers; grants nothing by itself

agents:
  databot:
    type: interactive
    max_actions: [read, write, generate]
    allow: ["read:db:sales.*", "read:web:*", "write:fs:reports/*", "generate:model:*"]
    deny: ["egress:*"]
    approvers: [ops-team]
  nightly_etl:
    type: autonomous                        # principal is svc:nightly_etl, U = this allow list
    max_actions: [read, write, generate]
    allow: ["read:db:sales.*", "read:web:*", "write:fs:reports/*", "generate:model:qwen3:8b"]
    approvers: [ops-team]                   # must be a role defined above

risk:
  half_life_s: 600

risk_rules:                                 # profile-independent session restrictions
  interactive:
    - when: { taint: true }
      then: { deny_actions: [write, delete, egress] }
    - when: { risk_gt: 0.5 }
      then: { actions: [write], mode: require_approval, cooldown_s: 300 }
    - when: { risk_gt: 0.8 }
      then: { freeze_tools: true, duration_s: 300 }
  autonomous:
    - when: { taint: true }
      then: { actions: [write, delete, egress], mode: require_approval }
    - when: { risk_gt: 0.5 }
      then: { throttle: { max_actions: 1, per_s: 10 }, alert: true }
    - when: { risk_gt: 0.8 }
      then: { actions: [write, delete, execute, egress], mode: require_approval, alert: true }

approvals: { timeout_s: 600, on_timeout: deny }

sessions: { idle_ttl_s: 3600, max_lifetime_s: 86400 }

throttle: { backoff: exponential, base_s: 5, max_s: 300 }

blocklist:                 # manual blocks, effective right after reload
  users: []
  agents: []
  use_cases: []            # permission patterns, e.g. "egress:http:pastebin.com"

limits:
  max_request_bytes: 1048576
  max_response_bytes: 4194304
  upstream_timeout_s: 120

controls:
  model_allowlist:  { mode: block }
  pii:              { mode: redact, threshold: 0.6,
                      entities: [PL_PESEL, PL_NIP, IBAN_CODE, EMAIL_ADDRESS, PHONE_NUMBER] }
  secrets:          { mode: block }
  prompt_injection: { mode: block, threshold: 0.85, judge_band: [0.5, 0.85], risk_delta: 0.6 }
  sql_guard:        { mode: block, max_cost: 10000, force_limit: 500, timeout_ms: 3000 }
  signatures:       { mode: block, feed: feeds/signatures.json, refresh_s: 30 }   # path relative to this file, or an http(s) URL
  loop_detect:      { mode: block, max_repeats: 5, window_s: 60 }

pricing:                   # USD; local models are priced by GPU wall-time estimate
  "qwen3:8b": { prompt_per_1k: 0.0, completion_per_1k: 0.0, gpu_second: 0.0005 }

budgets:
  per_user:    { daily_tokens: 200000, daily_cost_usd: 2.0 }
  per_agent:   { daily_tokens: 1000000 }
  per_session: { tool_calls: 50, gpu_seconds: 900 }   # CPU inference: ~10-70 s per call
  soft_limit_pct: 80
```

**Schema rules** (Pydantic, `extra="forbid"` everywhere):

- Loaded with a safe YAML loader that rejects duplicate keys; size capped at 1 MiB.
- Actions, modes, profiles, agent types, adapters and trust levels are enums. Every permission string is validated against the grammar above.
- Cross-references are checked: approver roles exist, every MCP tool maps to a known action, resource templates only use `{arg}` placeholders, `judge_band` is ordered and inside `[0, 1]`, durations and budgets are positive, risk thresholds are in `[0, 1]`.
- Mandatory controls (`authn`, `authz`, `secrets`, `sql_guard`) cannot be given `mode: log_only` and are active even when omitted from `controls`.
- Each control declares its supported modes; a configured mode it does not support is a validation error.
- `schema_version` is the file format; the **policy revision** is the SHA-256 of the canonicalized document (first 12 hex chars in audit). The signature feed has its own `version`.

**Strictness profiles** set the default mode of every non-mandatory detection control; a mode set on a specific control always wins. Profiles never touch `risk_rules`, base grants, or mandatory controls:

| Profile | Detection control default mode | Mandatory controls | No grant for a resource | Risk rules |
| --- | --- | --- | --- | --- |
| strict | most enforcing supported mode | enforce | deny | as configured |
| balanced | `redact` if supported, else `require_approval` if supported, else most enforcing | enforce | deny | as configured |
| permissive | least enforcing supported mode (`log_only` where supported) | enforce | deny | as configured |

Each control declares its supported modes ordered from most to least enforcing (`block` > `require_approval` > `redact` > `log_only`), which is how a profile resolves; e.g. `intent_judge` gets `require_approval` under strict and `log_only` under permissive, and `tool_pinning` stays `block` everywhere. Default deny applies in every profile. A profile changes how detections are handled, but never opens resources outside the allowlist.

**Hot reload:** a file watcher (`watchfiles`) and `POST /admin/reload` (admin token). Reload parses and validates the whole file, then atomically swaps an immutable snapshot; an invalid file is logged, counted (`acl_policy_reloads_total{result="invalid"}`) and never replaces the working one. The gateway refuses to start without a valid policy. Pending approvals stay bound to the revision they were created under and are re-checked against the current revision when consumed; a kill switch takes effect for in-flight calls at their next pipeline step. Every change gets a revision hash that goes into every audit entry and into a Grafana annotation, so you can see exactly when the new policy took effect.

## Audit, metrics and Grafana

Every decision leaves one audit entry with the full "why", and metrics feed dashboards for both security and management.

**Audit entry** (JSON to stdout, plus append-only JSONL segments `audit-<UTC time>-<n>.jsonl` that are never renamed, so Alloy ships every line to Loki even across an outage; the newest segments double as the export):

```json
{"ts":"2026-10-04T10:12:03Z","session_id":"s-81f","principal":"intern@demo","actor":"databot",
 "channel":"mcp","action":"read","resource":"db:sales.customers","decision":"block",
 "verdicts":[{"control":"authz","decision":"block","enforced":true,"reason_code":"resource_outside_scope"}],
 "effective_scope":["read:db:sales.orders"],"risk":0.35,"taint":false,
 "policy_revision":"a1c9e2f04b7d","feed_version":"2026-10-03.1",
 "latency_ms":{"total":7.4,"controls":{"authz":0.2,"pii":3.1}}}
```

Payloads, SQL text, tool arguments and upstream errors are never logged. An entry carries structured reason codes and metadata only; a keyed HMAC of the payload (not a plain hash, which is guessable for low-entropy inputs) lets an operator match an entry to a known payload. The JSONL export lives on an operator-only volume with a retention limit.

**Prometheus metrics** (`/metrics`):

- `acl_requests_total{channel,decision,agent}`
- `acl_control_verdicts_total{control,decision}`
- `acl_control_latency_seconds{control}` (histogram) and `acl_overhead_seconds{channel}` (gateway overhead excluding upstream)
- `acl_tokens_total{user,agent,model}`, `acl_cost_usd_total{user,agent,model}`
- `acl_budget_usage_ratio{scope,id}`
- `acl_session_risk` (histogram over sessions; per-session risk lives in Loki, never as a label), `acl_tainted_sessions`
- `acl_approvals_pending`, `acl_throttled_total{agent}`
- `acl_policy_reloads_total{result}`, `acl_policy_info{revision}` (gauge set to 1 for the active revision)

`user` and `agent` labels are bounded by the identities defined in the policy; unknown values are bucketed as `other`. `/metrics` is served on the operator network only.

| Dashboard | Audience | Panels |
| --- | --- | --- |
| Posture | management | Block rate, top threats, cost per team/agent, budget usage, daily trend |
| Threats | security | Stream of blocks from Loki, top controls, highest-risk sessions, signature hits, approval queue, policy change annotations |
| Session trace | security | Timeline of one session: decisions, effective permissions, risk score (`session_id` variable) |
| Performance | judges, ops | p50/p95/p99 overhead per channel and per control, throughput |

Dashboards and data sources are provisioned as files in the repo, so `docker compose up` brings up a ready Grafana. Prometheus scrapes the gateway's operator listener on the `ops` network. Audit entries reach Loki without the Docker socket: the gateway writes them to a JSONL file on a volume, and Grafana Alloy tails that volume read-only and pushes to Loki. Grafana, Prometheus, Loki and Alloy run only on `ops`; only Grafana is published, on `127.0.0.1` (default port 3300, since 3000 is commonly taken). Policy reloads appear as dashboard annotations from the audit/reload log lines in Loki.

## Test suite and demo

The competition rules weight the test suite at 20%, so every control has at least one allow/deny pair and the whole suite runs with one command: `make test`.

**Test layers:**

- `tests/unit/`: each control on its own, with a case table (`pytest.mark.parametrize`): input → expected verdict.
- `tests/policy/`: role × agent × action × resource matrix → allow/deny. Same request, different user, different result. Resources and tools outside the policy are denied in every profile (default deny).
- `tests/session_modes/`: the same event (taint, high risk) removes actions in an interactive session, while an autonomous session gets throttling and the approval queue, without losing rights.
- `tests/approvals/`: `require_approval` → queue → approve, deny, timeout; replay of a consumed approval, an approval with altered arguments, self-approval, revocation by a policy reload; manual kill switch and blocklist take effect after reload.
- `tests/bypass/`: run from inside the agent container, a direct request to Ollama, the router, MCP servers or Postgres does not get through; the fetch tool cannot reach internal addresses (SSRF).
- `tests/identity/`: forged, expired, wrong-audience and wrong-algorithm tokens; an unregistered agent; `mode` not matching registration; session reuse with a different principal.
- `tests/e2e/`: real requests through the gateway to Ollama and MCP servers in docker compose.
- `tests/attacks/`: a corpus of attack prompts and payloads (direct and indirect injection, secret leakage, PII exfiltration, tool poisoning, SQL attacks, path traversal, SSRF, approval and identity abuse; pickle scanning is out of scope) with the expected block.
- `tests/budget/`: token and call budget exhaustion, loop detection, throttling with backoff.
- `tests/reload/`: changing `policy.yaml` mid-test changes the verdict without restart; a broken file does not break the working policy.
- `tests/perf/`: p50/p95 overhead with a mocked upstream, written into the report.

Output: JUnit XML + HTML report (`pytest-html`) summarizing positive and negative cases per control. Tests declare what they prove with a `control(<id>, outcome)` marker; a coverage check fails the run if any catalog control lacks an allow and a deny case, and the per-control table is written to `reports/controls.md` and embedded in the HTML report. `tests/perf/` writes `reports/perf.json` and `reports/perf.md`. Tests are written alongside each stage, not saved for the end: every stage in the schedule ends with its own passing tests.

**Demo script (3 minutes):**

1. Anna (analyst) and Bartek (intern) send DataBot the same prompt asking for the customer count. Both may read `sales.customers`; RLS returns a different `COUNT(*)` for each (seeded so the numbers clearly differ). Bartek asking for `sales.payments` is blocked at the table level.
2. DataBot writes a report (allowed: `write:fs:reports/*`). It then reads a page with hidden injection through the `web` tool. The session is tainted, the same report write is now blocked, and the risk score rises in Grafana.
3. The same injection hits the autonomous `nightly_etl`. The write is not blocked; it waits in the approval queue, and the process slows down instead of stopping. An operator approves it and the exact call goes through once.
4. The agent generates a heavy query. `sql_guard` rejects it after `EXPLAIN`.
5. A prompt with a PESEL number is answered with the PESEL redacted. A second prompt with an API key is blocked. (On CPU the output_policy judge can time out and hold the answer back; the redaction still shows in the audit.)
6. From inside the agent container, a direct call to Ollama does not connect.
7. A judge raises `max_cost` in `policy.yaml` or adds a signature to the feed. The same request gets a different verdict, and the audit entry shows the new policy revision or feed version; a policy change also shows as a Grafana annotation.

## Stack, repo layout and schedule

Python 3.12 throughout, managed with `uv`; `ruff` (lint + format) and `pyright` gate every change; Pydantic models for every boundary (policy, tokens, envelopes, API payloads). Everything local in `docker compose`, no paid APIs. We check the license of every dependency before using it and pin exact versions in `uv.lock`; images and models are pinned by tag/digest. "No paid dependencies" is not "all permissively licensed": Grafana and Loki are AGPLv3 and Redis ≥ 7.4 is RSALv2/SSPLv1 (≥ 8.0 adds AGPLv3), so we use `redis:7.2` (BSD-3) or Valkey and record the terms in `LICENSES.md`. The LiteLLM package mixes MIT code with an `enterprise/` directory under a separate license; we import only its MIT cost tables. Presidio is MIT, but its spaCy models are reviewed separately and the Polish recognizers (PESEL, NIP) are ours.

| Layer | Choice | Notes |
| --- | --- | --- |
| Proxy | FastAPI + httpx, uvicorn | Buffered upstream calls; SSE re-emitted to the agent after post controls |
| LLM client / cost | `litellm` as a library | We do not use LiteLLM Proxy |
| MCP | official MCP Python SDK | Proxy acts as an MCP server to the agent and a client to upstreams |
| Policies | Pydantic + our own evaluator | Option: Cedar (`cedarpy`) if time allows |
| PII | Presidio + Polish regexes |  |
| SQL | `sqlglot`, Postgres with RLS |  |
| Injection classifier | small Hugging Face model on CPU, run with ONNX Runtime + `tokenizers` (no torch) | Candidate `protectai/deberta-v3-base-prompt-injection-v2` (Apache-2.0, English-only, 512-token window): pin the revision and the SHA-256 of every file, load the ONNX export only (no pickle, no `trust_remote_code`), chunk long inputs, and test Polish inputs before relying on it. Weights are fetched by a `models-init` compose profile into a read-only volume, the same pattern as `ollama-init`; the gateway refuses to start with `prompt_injection` enabled and no verified model |
| LLM judge and demo models | Ollama (e.g. Qwen3 8B) |  |
| Session state, budgets, approvals | Redis |  |
| Telemetry | prometheus-client, Loki, Grafana | Provisioned from the repo |
| Demo agent | simple Python agent with MCP | Not assessed |

```
ai-control-layer/
  gateway/
    main.py            # FastAPI app factory, routers: /v1 (LLM), /mcp/{server}, /auth, /admin
    core/              # Interaction, Verdict, Span, SessionContext, Adapter, Control, verdict merging
    pipeline.py        # step order
    identity.py        # JWT verification, delegation checks, demo token issuer
    sessions.py        # session store ABC + in-memory store; redis_sessions.py: Redis store, distributed lock
    policy/            # schema, permission grammar, loader, hot reload, evaluator
    proxies/           # llm.py, mcp.py
    adapters/          # generic_mcp, sql, http, fs, llm
    controls/          # one class = one file
    approvals/         # approval queue, kill switch, operator API (throttle.py: throttling)
    telemetry.py       # metrics, audit
  config/            # mounted read-only as a directory so hot reload sees editor saves
    policy.yaml
    feeds/signatures.json
  demo/                # agent, identities, database seed, MCP servers
  grafana/             # dashboards/, provisioning/
  tests/
  docker-compose.yml   # edge / ops / llm_backend / mcp_backend / mcp_untrusted / fetch_egress / state / bootstrap networks
  pyproject.toml       # uv, ruff, pyright, pytest config
  Makefile             # up, test, lint, demo, report
```

**Schedule (hours from now, 22h):**

1. 0–5: gateway skeleton, core interfaces, policy loader with default deny and hot reload, JWT and session state, LLM proxy to Ollama (non-streaming upstream, SSE re-emission), generic MCP proxy (tools-only subset, operator-mapped tools), upstream isolation across compose networks. Each step ends with its own pytest tests; the compose isolation test is marked `docker` and runs only against a live stack.
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

The honest pitch line: we do not claim to have invented taint or on-behalf-of, and a feature missing from a vendor's public docs is not proof it does not exist. We show that combining them in one layer addresses the agent permission problem the challenge owner called the biggest one.

**Sources** (as of 2026-10-03): links in the table.

## Changelog

**v2 (2026-10-03), after an external review of v1:**

- Fixed `policy.yaml`: v1's `risk_rules` lines put `when:` and `then:` on one line, which is not valid YAML.
- Replaced the single `E = U ∩ A ∩ T \ R` formula with base grants plus session restrictions and obligations; an approval never grants anything outside base grants.
- Autonomous agents get a service principal (`svc:<agent>`) and explicit `allow` grants; session mode is a token claim validated against the agent's registration.
- Defined the permission grammar (split at the first colon, `*` only, anchored) and evaluate grants per concrete interaction.
- Profiles affect only detection controls; `authn`, `authz`, `secrets`, `sql_guard` are mandatory and always enforce; `log_only` is `enforced=false` on a verdict, not a decision.
- `Control.stages` (pre, post or both); verdict merging keeps obligations (approval + redaction); requests execute once after all envelopes pass.
- JWT hardening (pinned algorithm, `iss`/`aud`, lifetime cap, session binding), a demo token issuer restricted to predefined identities, admin auth.
- Split compose networks (edge, llm_backend, mcp_backend, mcp_untrusted); bypass tests run from inside the agent container, plus SSRF.
- MCP: operator-owned tool mappings (annotations are hints only), filtered `tools/list`, pinned protocol version, tools-only subset, no shared upstream sessions.
- `intent_judge` is advisory; every MCP call is authorized on its own.
- Approvals bind one exact operation (argument digest, policy revision) with explicit states, including `uncertain`.
- RLS: the trusted SQL MCP server owns the transaction and uses transaction-local `set_config`; non-owner role, `FORCE ROW LEVEL SECURITY`.
- `sql_guard`: supported-subset allowlist, `EXPLAIN (FORMAT JSON)` without `ANALYZE`, timeouts, and the exact rewritten query is what runs.
- Streaming: upstream calls are buffered; SSE is re-emitted after post controls.
- Budget reservation before dispatch; fail closed without Redis; per-session serialization; immutable policy snapshots; refuse to start without a valid policy.
- Audit: reason codes and keyed HMACs instead of excerpts; no per-session metric labels.
- Demo repaired so restricted actions are visibly allowed before taint, RLS shows different counts on the same table, and PII and secrets are separate requests.
- Licensing notes for Redis, Grafana/Loki, LiteLLM, Presidio and the injection classifier; pickle scanning removed from scope.
- Second review pass: whole-resource `*` wildcard; profile resolution by each control's ordered supported modes; `Verdict.rewrite` for payload rewrites; MCP schema provenance and `tools/list` eligibility; DataBot approvers and approver-scoped `/admin`; `intent_judge` flags the later MCP call instead of holding the LLM response; cooldown vs freeze vs throttle timers; `ops` network and two gateway listeners; `/v1/models` filtering; session end and TTLs; per-control `risk_delta` defaults and taint on blocked results.
- Stage 5–10 decisions: `EXPLAIN` runs in the SQL MCP server through a gateway-only `explain` tool; execution limits travel in the signed `X-ACL-Principal` assertion; budgets use a policy `pricing` table instead of a `litellm` dependency; Redis joins on its own `state` network.
- After the stage 5–10 review: `acl.set_principal` replaces a bare `set_config`; `Interaction.server`; all CTEs refused; `sql_guard` seals the final SQL; feed is a file path; GPU allowance as an upstream deadline; idempotent budget operations.
- Stage 10–14 decisions: the injection classifier runs on ONNX Runtime and is fetched by a `models-init` profile with pinned hashes; one shared `JudgeClient` for all LLM judges, failing closed and not charged to agents; approvals and the kill switch live in Redis; session state and loop counters move to Redis behind the existing interfaces.
- After the stage 10–14 review: operator tokens on their own audience; `sid_iat` and issuer-minted session ids; approval binding covers resources and the final operation digest, kill voids approvals by generation; tool quarantine until re-approval; prose-only, HTML-aware classification; strict judge response models; a session whose outcome could not be persisted stays fenced and resolves as tainted.
- Stage 14–18 decisions: Loki ingestion via Alloy tailing the audit JSONL volume (no Docker socket); observability services on `ops` only, Grafana on 127.0.0.1:3300; `control(<id>, outcome)` test markers drive the per-control report and coverage check.
- After the first real-model run: judges send `reasoning_effort: none`; Ollama keeps two cache slots; per-session GPU budget 900 s for CPU inference; classifier hits on agent-authored prompts are judge-confirmed and taint on timeout instead of blocking outright.
- Stage 18–22: the demo overlay's exact-name egress/fetch exemptions (off by default, audited); spec layout and network list synced with the code.
