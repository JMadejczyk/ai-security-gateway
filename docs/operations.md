# Operations

Day-two tasks for operators: tokens for the admin API, the human-approval queue, the kill switch, and reviewing MCP tool baselines.

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

Every MCP server needs a pin file in `pins/` (`require_pin: true` in `config/policy.yaml`); a tool that
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
