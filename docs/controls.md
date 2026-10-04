# The 15 controls

Every call runs through a hybrid set of guardrails. **Rule-based** (deterministic) controls are
cheap and run on every call: 1.6 ms p95 combined on a typical prompt in the
[benchmark](reports/perf.md). **AI-based** (semantic) controls run in tiers: a local
classifier on every input, and an LLM judge only when the classifier is uncertain, or to
confirm a hit on text a person wrote. Each control's mode (block, redact, require approval,
log only) and thresholds are set in the [policy file](policy.md); *mandatory* controls
enforce in every profile.

| ID | Type | Stage | Channel | What it does | Modes |
| --- | --- | --- | --- | --- | --- |
| `authn` | rule | pre | all | Verify JWT, agent and role exist, session binding (mandatory) | block |
| `authz` | rule | pre | all | Base authorization + session restrictions vs `action:resource` (mandatory) | block, require\_approval |
| `model_allowlist` | rule | pre+post | llm | Allowed models per role/agent; also checks the model in the upstream response | block, log\_only |
| `pii` | rule | pre+post | all | Presidio + Polish regexes (PESEL, NIP, IBAN) | redact, block, log\_only |
| `secrets` | rule | pre+post | all | API keys, tokens, private keys, connection strings (mandatory) | redact, block |
| `sql_guard` | rule | pre | mcp (SQL) | Parse with `sqlglot` (Postgres dialect), allow only a supported `SELECT` subset (single statement, no DDL/DML, no CTEs, schema-qualified tables only, no `COPY`, no session or settings functions, only allowlisted functions, every table resolved), `EXPLAIN (FORMAT JSON)` without `ANALYZE` against a planner-cost threshold, forced `LIMIT`, `statement_timeout`/`lock_timeout`, result-byte limit; the exact rewritten query is what executes (mandatory) | block |
| `egress` | rule | pre | mcp (HTTP) | Parsed scheme/host/port allowlist, DNS resolution checked against private, loopback and metadata ranges, redirects re-validated; taint handling comes from `risk_rules` per session mode | block, require\_approval |
| `signatures` | rule | pre+post | all | Match against an external attack signature feed (size-bounded, schema-validated, regex with a timeout, last valid feed kept) | block, log\_only |
| `tool_pinning` | rule | pre | mcp | Pin tool names, input schemas, descriptions and annotations to the approved baseline (`pins/<server>.json`, reviewed with `acl pin`); servers have `require_pin: true` by default. Any change (rug pull) quarantines the tool in Redis across sessions until an operator re-pins or clears it. Does not prove server implementation integrity | block |
| `budget` | rule | pre+post | all | Tokens, cost, tool calls, GPU time per user/agent/session | block |
| `loop_detect` | rule | pre | all | Same call N times within window T | block |
| `prompt_injection` | AI | pre+post | llm, mcp | Classifier on prompts and on tool results (indirect injection) | block, log\_only |
| `tool_poisoning` | AI | pre | mcp | Classifier on MCP tool descriptions at registration | block |
| `intent_judge` | AI | post | llm | LLM judge compares a `tool_call` with the user's goal | require\_approval, log\_only |
| `output_policy` | AI | post | llm | Does the answer reveal data outside the session scope | redact, block |

**Stage:** `pre` runs before the upstream is called, `post` on its answer.
**Channel:** `llm` is the `/v1` model proxy, `mcp` the `/mcp` tool proxy.

## What a detection does next

A detection doesn't only decide one call. Each verdict adds its `risk_delta` to the session's
risk, and some set **taint**. The reaction then depends on who is driving:

| Session state | Interactive session (human) | Autonomous system (registered) |
| --- | --- | --- |
| `taint = true` | `write`, `delete`, `egress` removed until session end | `write`, `delete`, `egress` need human approval |
| `risk > 0.5` | `write` needs approval; a denied call starts a 5 min cooldown | Throttling with exponential backoff + alert |
| `risk > 0.8` | Tool calls frozen for 5 min, `generate` still allowed | Everything except `read` and `generate` goes to the approval queue |

Risk decays over time (half-life 10 min by default); taint lasts until the session ends.
Automation never permanently revokes rights granted in the policy: it can slow down, hold for
approval, or freeze a human's session for a few minutes. Permanently disabling an agent (the
kill switch) is always a human decision.

## Proof

Every control has passing *allow* and *deny-side* tests; the build fails otherwise. See the
[control coverage](reports/controls.md) table and the [test suite](testing.md). The
[full specification](spec.md#control-catalog) describes each control's edge cases
(streaming, classifier scope, judges, intent vs enforcement).
