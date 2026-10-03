# AI Control Layer

A security gateway that all agent traffic passes through: to models (an OpenAI-compatible
`/v1` proxy) and to tools (an MCP proxy). Each call is authorized on the intersection of the
person's rights, the agent's rights and the task scope. Threat detections raise session risk
and taint, and those change what the agent may do for the rest of the session. An interactive
session loses actions; an autonomous system gets throttled and sent to human approval.
`SPEC.md` is the source of truth.

## Layout

| Path | What lives there |
| --- | --- |
| `gateway/core/` | Shared vocabulary: `Interaction`, `Verdict`, `Span`, `SessionContext`, the `Adapter` and `Control` interfaces, verdict merging, the control catalog |
| `gateway/policy/` | Permission grammar, `policy.yaml` schema, `PolicyLoader`, the hot-reloading `PolicyStore`, the `PolicyEvaluator` (decision model) |
| `gateway/identity.py` | JWT verification and delegation checks (`TokenVerifier`), the demo token issuer, `X-ACL-Principal` minting |
| `gateway/sessions.py`, `gateway/redis_sessions.py` | `SessionStore`: binding, lifetimes, per-session lock, risk decay, sticky taint; Redis-backed in compose (`ACL_SESSION_STORE`), so state survives restarts and is shared by gateways |
| `pins/` | Operator-approved MCP tool baselines for `tool_pinning`, one `<server>.json` per server, written by `python -m gateway.cli pin` |
| `gateway/pipeline.py` | The step order every call goes through: authn, adapter, authz, pre controls, merge, upstream, post controls, persist, audit |
| `gateway/adapters/`, `gateway/proxies/` | Per-channel normalization (`LLMAdapter`) and upstream clients (`LLMProxy`, SSE re-emission) |
| `gateway/controls/` | The `ControlRegistry` new guardrails register with |
| `gateway/main.py`, `gateway/__main__.py` | The agent and operator FastAPI apps; `python -m gateway` serves both |
| `gateway/telemetry.py` | Prometheus metrics (a dedicated registry) and the JSON audit log |
| `policy.yaml` | The policy the gateway runs with; it is validated, and reloaded on change |
| `tests/` | `unit/`, `policy/`, `session_modes/`, `reload/`, `identity/`, `sessions/`, `llm/` (plus docker-marked suites) |

## Develop

Needs [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned and uv fetches it if missing.

```sh
make install   # uv sync --frozen
make lint      # ruff check, ruff format --check, pyright (strict on gateway/)
make fmt       # ruff --fix + ruff format
make test      # pytest without docker-marked tests; reports/junit.xml + reports/report.html
make up        # docker compose up -d --build
make down      # docker compose down
```

Run the gateway outside compose (two listeners: agent API on 127.0.0.1:8080, operator API on
127.0.0.1:9090; both overridable with `ACL_AGENT_HOST`/`_PORT` and `ACL_OPERATOR_HOST`/`_PORT`):

```sh
ACL_JWT_SECRET=... ACL_INTERNAL_KEY=... uv run python -m gateway   # both at least 32 bytes
curl -s -X POST localhost:9090/auth/demo-token -H 'content-type: application/json' \
  -d '{"sub": "anna@demo"}'                                         # ACL_DEMO_TOKENS=0 disables
```

`ACL_POLICY_PATH` and `ACL_IDENTITIES_PATH` default to `policy.yaml` and `demo/identities.yaml`;
`ACL_AUDIT_PATH` adds a JSONL export next to the audit lines on stdout.

The gateway refuses to start without a valid policy or strong secrets. A policy edit that
fails validation is logged and counted (`acl_policy_reloads_total{result="invalid"}`), and the last valid
version stays in force.

## Operator tokens, approvals and the kill switch

`/admin/*` on the operator listener takes **operator tokens** only: audience
`ai-control-layer-operator`, no agent and no session. Agent tokens (even an admin's) get 401
`wrong_audience` there, and operator tokens get the same on the agent listener. The demo issuer
mints one for identities holding `admin` or an approver role (`olga@demo`, `root@demo`):

```sh
TOKEN=$(curl -s -X POST 127.0.0.1:9090/auth/demo-token -H 'content-type: application/json' \
  -d '{"sub": "olga@demo", "kind": "operator"}' | jq -r .access_token)
export ACL_OPERATOR_TOKEN=$TOKEN
uv run acl approvals list                       # pending approvals olga may decide
uv run acl approvals show apr-...               # tool, resources, reasons, digests: no arguments
uv run acl approvals approve apr-... --note ok  # or: deny
uv run acl kill nightly_etl --reason incident   # admin only; `acl unkill nightly_etl`
```

An approver sees and decides the approvals of agents whose `approvers` name one of their roles
(`admin`: all), never one of their own sessions. A held MCP call answers a tool error carrying
`_meta["ai-control-layer/approval_id"]`; once approved, the agent retries the same `tools/call`
with `_meta: {"ai-control-layer/approval_id": "<id>"}` in its params (LLM: the
`X-ACL-Approval-Id` header). The retry is re-authorized under the current policy and runs at
most once. `acl` is the same CLI as `python -m gateway.cli`.

## Approving MCP tool baselines (`tool_pinning`)

Every MCP server needs a pin file in `pins/` (`require_pin: true` in `policy.yaml`); a tool that
differs from its baseline (description, input schema, annotations) or is not in it is hidden
from `tools/list` and refused on `tools/call` until it is re-approved. Review and approve on the
host, against the running stack (admin token; `pins/` is mounted read-only into the gateway and
a new file applies on the next call):

```sh
TOKEN=$(curl -s -X POST 127.0.0.1:9090/auth/demo-token -H 'content-type: application/json' \
  -d '{"sub": "root@demo", "kind": "operator"}' | jq -r .access_token)
ACL_OPERATOR_TOKEN=$TOKEN uv run python -m gateway.cli pin sales_db          # diff, exit 3 if changed
ACL_OPERATOR_TOKEN=$TOKEN uv run python -m gateway.cli pin sales_db --write  # approve
```

A tool caught drifting is quarantined for every session and gateway (listed by the review)
until an operator re-approves a new baseline with `--write`, or confirms the server is back
to the approved definition and runs `pin <server> --clear-quarantine <tool>`.

The same review runs inside the gateway container, which sees the upstreams directly:
`docker compose exec -e ACL_OPERATOR_TOKEN gateway sh -c 'python -m gateway.cli --url
"http://$ACL_OPERATOR_HOST:$ACL_OPERATOR_PORT" pin sales_db'` (read-only there: no `--write`).
