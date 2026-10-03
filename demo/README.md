# Demo stack

`docker-compose.yml` at the repo root runs the gateway with its demo upstreams. Everything in
this directory belongs to that stack:

| Path | What it is |
| --- | --- |
| `identities.yaml` | The principals `POST /auth/demo-token` may issue tokens for, with each one's roles, mode and default agent |
| `db/` | Postgres init: the `sales` schema, roles, row-level security and a small seed |
| `mcp_servers/` | One image that serves the three MCP upstreams: `postgres`, `fetch`, `files` |
| `agent/` | The agent container: it can reach only the gateway. `acl_agent/` is DataBot, the demo agent (httpx: MCP + OpenAI chat), run one action at a time with `docker compose exec` |
| `compose.demo.yml`, `web/` | The demo overlay: `demo-web` serves the page with the hidden injection to mcp-fetch only, plus the two named egress exceptions it needs (see `docs/demo.md`) |
| `run_demo.py`, `orchestrator/` | The host-side demo orchestrator (`make demo`) |

## Running it

```sh
cp .env.example .env                                # replace every secret in .env
docker compose --profile init run --rm ollama-init  # one-off model pull (OLLAMA_MODEL, several GB)
docker compose up -d --build
```

`ollama-init` is the only way a model gets into the `ollama_models` volume. It runs on the
temporary `bootstrap` network, which has internet access. The `ollama` service itself sits on
`llm_backend` only, so it has no way out. Pull the model before `up`, or stop `ollama` while
pulling, so two processes don't write the volume at once.

`config/` (the policy and the signature feed) is mounted into the gateway as a read-only
directory, so editing `config/policy.yaml` on the host takes effect without a restart: the
gateway's watcher reloads it within a few seconds (a Grafana annotation marks the new
revision). A single-file bind mount would not work: editors and `git checkout` replace the
file with a new inode, and the container would keep seeing the old one.

If 9090, 8080 or 3300 is already taken on your machine, set `ACL_OPERATOR_HOST_PORT`,
`ACL_AGENT_HOST_PORT` or `ACL_GRAFANA_HOST_PORT` in `.env`. All three are published on
`127.0.0.1` only.

## Network topology

```mermaid
flowchart LR
    agent[agent] -- edge --> gw8080[gateway :8080]
    host[(host 127.0.0.1)] --> gw9090[gateway :9090<br/>ops IP only]
    host --> gw8080
    gw8080 -. same process .- gw9090
    gw8080 -- llm_backend --> ollama
    gw8080 -- mcp_backend --> mcppg[mcp-postgres] --> postgres
    gw8080 -- mcp_backend --> mcpfiles[mcp-files]
    gw8080 -- mcp_untrusted --> mcpfetch[mcp-fetch] -- fetch_egress --> internet((internet))
    gw8080 -- state --> redis[(redis)]
    prometheus -- ops: scrape /metrics --> gw9090
    gw9090 -. audit-*.jsonl on audit_log volume .-> alloy -- ops --> loki
    grafana -- ops --> prometheus
    grafana -- ops --> loki
    host --> grafana[grafana :3000<br/>127.0.0.1:3300]
```

| Network | Internal | Members |
| --- | --- | --- |
| `edge` | yes | agent, gateway |
| `ops` | no | gateway (static `172.29.90.10`), prometheus, loki, alloy, grafana |
| `llm_backend` | yes | gateway, ollama |
| `mcp_backend` | yes | gateway, mcp-postgres, mcp-files, postgres |
| `mcp_untrusted` | yes | gateway, mcp-fetch |
| `state` | yes | gateway, redis |
| `fetch_egress` | no | mcp-fetch only |
| `demo_web` | yes | mcp-fetch, demo-web (only with the overlay `demo/compose.demo.yml`) |
| `bootstrap` | no | ollama-init, models-init (profile `init`) |

Two choices are worth spelling out:

- **The operator listener binds the gateway's `ops` IP, not `0.0.0.0`.** A socket on
  `0.0.0.0` answers on every network the container joins. The agent on `edge` could then call
  `/auth/demo-token`. The gateway reads its bind addresses from `ACL_AGENT_HOST`/`ACL_AGENT_PORT`
  and `ACL_OPERATOR_HOST`/`ACL_OPERATOR_PORT`, and compose sets the operator host to the static
  `ops` address.
- **`mcp_untrusted` is internal, and mcp-fetch reaches the internet through `fetch_egress`.**
  Docker sends a container's published ports to its default-route network. That has to be
  `ops`, the gateway's only non-internal network. If `mcp_untrusted` carried the internet route,
  Docker could pick it instead, and the operator listener would not be reachable from the host.
  The property the SPEC asks for still holds: only the fetch server has internet access, and it
  shares no network with Postgres or Ollama.

`tests/bypass/` checks all of this. The static tests read `docker compose config`. The
`docker`-marked tests run against a running stack. They send TCP probes from inside the agent and
mcp-fetch containers to every protected container's real IP on every network it joins, with
positive controls from the gateway. They also call the `fetch` tool with internal URLs:

```sh
pytest tests/bypass -m "not docker"         # static, needs only the docker CLI
ACL_DOCKER_TESTS=1 pytest tests/bypass      # also the live probes (stack must be up)
```

## Observability (`prometheus`, `loki`, `alloy`, `grafana`)

All four join `ops` and nothing else, so neither the agent (`edge`) nor mcp-fetch
(`mcp_untrusted`, `fetch_egress`) can resolve or reach them. They run as their images'
non-root users (Alloy as its `alloy` user, uid 473, instead of the image default root), with a
read-only root filesystem, `cap_drop: ALL` and `no-new-privileges`. Their images are pinned to
exact versions (`LICENSES.md` records them; Grafana and Loki are AGPLv3, used unmodified).

- `prometheus` scrapes `gateway:9090/metrics`. It shares only `ops` with the gateway, so the
  name resolves to the operator listener's static address.
- The gateway writes its audit JSONL to the `audit_log` volume. `alloy` mounts that volume
  read-only, tails the `audit-*.jsonl` segments (never renamed, so an Alloy outage loses
  nothing within the retention) and pushes to `loki`. Alloy gets no Docker socket, and its
  own HTTP server listens on 127.0.0.1 inside its container.
- `grafana` is the only one that publishes a port (`127.0.0.1:3300`). It is provisioned
  read-only from `grafana/`, and it is the only service holding `ACL_GRAFANA_ADMIN_PASSWORD`.
  The password applies when the `grafana_data` volume is first created.
- Grafana's analytics, update checks, news feed and plugin preinstall are off, and so are
  Loki's usage reports and Alloy's reporting. `ops` has an internet route because it carries
  the published ports, so these services never phone home.

`tests/bypass/` checks this too. Statically: networks, hardening, volumes, no socket, which
service holds which secret. Live: probes from the agent and mcp-fetch to every observability
address and name, with positive controls from the gateway.

## Budget counters (`redis`)

The gateway keeps budget counters (SPEC "Budgets") in `redis:7.2.16`, the last BSD-3 Redis
line. Redis sits on `state` alone with the gateway, publishes no port and requires the
password `ACL_REDIS_PASSWORD`, which only `redis` and `gateway` receive. The password reaches
`redis-server` through a `0600` config file on tmpfs, never its command line. The root
filesystem is read-only, and AOF on the `redis_data` volume keeps the day's counters across a
Redis restart.

If Redis is unreachable, every call a budget limits is refused with `503
budget_store_unavailable` (fail closed). `GET /healthz` still answers 200, with `"status":
"degraded"` and `"budget_store": "down"`, because the gateway is still the process serving
those refusals. Outside compose, `ACL_BUDGET_STORE=memory` keeps counters in the gateway
process for local development. The gateway never falls back to it by itself.

## Identities (`identities.yaml`)

```yaml
identities:
  <sub>:                      # JWT `sub`: user@demo, or svc:<agent> for autonomous agents
    roles: [<role>, ...]      # roles from policy.yaml
    mode: interactive | autonomous   # must equal the default agent's registered `type`
    default_agent: <agent>    # JWT `act.sub` unless the token request names another agent
    description: <text>       # a note for people; never used in a decision
```

| Principal | Role | Default agent | Customers visible (RLS) |
| --- | --- | --- | --- |
| `anna@demo` | analyst | databot | 40 (north, south, east) |
| `bartek@demo` | intern | databot | 7 (east) |
| `olga@demo` | ops-team (approver) | databot | 0 |
| `root@demo` | admin | databot | 50 |
| `svc:nightly_etl` | none (autonomous; grants come from the agent's `allow`) | nightly_etl | 50 |

`olga@demo` and `root@demo` can also get an **operator token** (`{"sub": ..., "kind":
"operator"}`): no agent, no session, accepted only by `/admin/*`. With it olga approves, say,
a databot write held in anna's session; nobody can approve a call of their own session, and
identities without `admin` or an approver role (anna, bartek, the service principal) get no
operator token at all (`not_an_operator`).

## Database (`db/`)

`00_init.sh` runs once, on an empty `pgdata` volume. It must stay executable (git mode 100755).
It applies `sql/01_schema.sql` and `sql/02_seed.sql`:

- `sales_owner` owns everything and cannot log in. `acl_app` is the role mcp-postgres connects
  as: not the owner, no `SUPERUSER`, no `BYPASSRLS`, `SELECT` only.
- `acl_app` cannot execute `set_config`. The only way to set `app.user_id` is
  `acl.set_principal(text)`, a `SECURITY DEFINER` function owned by the `NOLOGIN` role
  `acl_definer`, which refuses a second call in the same transaction. mcp-postgres calls it
  first, so a statement can change neither the principal nor any other setting.
- `sales.customers`, `sales.orders` and `sales.payments` have `ENABLE` + `FORCE ROW LEVEL
  SECURITY`. Which customers you see depends on `current_setting('app.user_id', true)`, matched
  against `acl.region_access`. Orders and payments follow the customers you can see. An unset
  user sees no rows.
- `acl.region_access` lives outside `sales`, so a `read:db:sales.*` grant never covers it.
- 50 customers, 1000 orders (20 per customer), 750 payments; the seed ends with `ANALYZE`, so
  planner estimates are stable. Planner costs (`sql_guard`, `max_cost: 10000`), as anna:
  `SELECT COUNT(*) FROM sales.customers` 2.88; `SELECT * FROM sales.orders` (capped at 500
  rows) 22.38; `customers CROSS JOIN orders CROSS JOIN payments` with `COUNT(*)` 70495.20,
  refused (demo step 4).

To re-seed: `docker compose down -v`, or remove the `pgdata` volume.

## MCP upstreams (`mcp_servers/`)

Each upstream is built on the official MCP Python SDK (`mcp==2.3.0`). In 2.x, `FastMCP` is
called `MCPServer`. Each one serves streamable HTTP on `0.0.0.0:8000/mcp`, and the compose
`command` picks which server runs.

| Service | Tool | Notes |
| --- | --- | --- |
| `mcp-postgres` | `query(sql)`, `explain(sql)` | Verifies `X-ACL-Principal` and its `limits`, checks the statement is one plain `SELECT`, then runs it in one read-only transaction that starts with `acl.set_principal(principal)` and `SET LOCAL statement_timeout` / `lock_timeout`. `query` runs exactly that statement through a server-side cursor (extended protocol: one statement), fetches at most `max_rows + 1` rows and refuses more rows or more than `max_result_bytes`. `explain` returns `EXPLAIN (FORMAT JSON)`'s top-level `Total Cost` without running the statement; it is called only by the gateway's `sql_guard` and is absent from the operator mapping, so agents never reach it. Without a valid principal and limits, nothing runs. |
| `mcp-files` | `write_report(name, content)` | Plain file names only. Creates files only (O_EXCL, so it never overwrites or follows a symlink). Writes under the `reports` volume at `/data/reports`. |
| `mcp-fetch` | `fetch(url)` | http(s) GET on port 80/443 only, with a 10 s timeout and a 256 KiB cap. It resolves the host itself and refuses the call if any answer is not a public address (loopback, private, CGNAT, link-local/metadata, ULA, multicast, and IPv4-mapped/6to4 forms of those). It then connects to the validated IP, keeping the original Host header and TLS SNI, so DNS rebinding can't redirect the connection. It does not follow redirects: the agent has to fetch the new URL through the gateway. |

Tool annotations (`readOnlyHint`, `destructiveHint`, ...) are only hints. The gateway's operator
mapping in `config/policy.yaml` decides what each tool means.

### `X-ACL-Principal`

The gateway sends this header to mcp-postgres on every call. It holds an HS256 JWT signed with
`ACL_INTERNAL_KEY`. Only the gateway and mcp-postgres have that key. It is not the agent signing
key, and it is never the agent's bearer token.

| Claim | Value |
| --- | --- |
| `iss` | `ai-control-layer` |
| `aud` | `mcp-postgres` |
| `sub` | the authenticated principal, e.g. `anna@demo` |
| `iat`, `exp` | `exp - iat <= 60` seconds |
| `limits` | `{stmt_timeout_ms, max_rows, max_result_bytes}` from `controls.sql_guard` (`timeout_ms`, `force_limit`, `max_result_bytes`); required |

The signature matters even on an internal network: mcp-files also sits on `mcp_backend`, and
without the key it cannot make the database act as another user, nor lift the limits.

### Developing the servers

```sh
cd demo/mcp_servers
uv run pytest                      # unit tests: report paths, principal, SQL checks, fetch SSRF
ACL_DOCKER_TESTS=1 uv run pytest   # + mcp-postgres against a throwaway Postgres seeded from db/
uv run pyright                     # strict
```
