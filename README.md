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
| `gateway/telemetry.py` | Prometheus metrics (a dedicated registry) |
| `policy.yaml` | The policy the gateway runs with; it is validated, and reloaded on change |
| `tests/` | `unit/`, `policy/`, `session_modes/`, `reload/` (plus docker-marked suites) |

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

The gateway refuses to start without a valid policy. A policy edit that fails validation
is logged and counted (`acl_policy_reloads_total{result="invalid"}`), and the last valid
version stays in force.
