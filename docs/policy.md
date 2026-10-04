# Policy file

One source of truth: `config/policy.yaml`. It is validated with a Pydantic schema and
**reloaded without a restart**. An invalid file never replaces a working one: the gateway logs
the error, counts it (`acl_policy_reloads_total{result="invalid"}`) and keeps the last valid
version. Every audit entry records the policy revision it was decided under, and each reload
puts a marker on the Grafana dashboards.

## What you can change

- **Profile and default:** `profile: strict | balanced | permissive`; `default: deny` is the
  only accepted value, so anything not granted is forbidden.
- **Roles and agents:** who may do what (`read:db:sales.*`, `write:fs:reports/*`, ...), per
  role and per agent; which principals may use an agent; approvers.
- **Controls:** each control's `mode` (block, redact, require approval, log only), thresholds
  such as `prompt_injection.threshold` and its judge band, `sql_guard.max_cost`, `loop_detect`
  limits. Mandatory controls reject `log_only`.
- **Risk rules:** what taint and each risk level do, separately for interactive and autonomous
  sessions.
- **Budgets and pricing:** per user, agent and session (tokens, USD, tool calls, GPU seconds).
- **Upstreams and models:** the local Ollama upstream and the opt-in OpenRouter upstream
  (zero-retention routing, price ceiling, model map), plus the judge model.
- **Signature feed:** `config/feeds/signatures.json` (or an http(s) URL), refreshed every
  `refresh_s`; add a signature and it applies without a restart.

!!! tip "Try it"
    With the stack running, raise `controls.sql_guard.max_cost` from `10000` to `100000`. A
    query that was refused with `sql_cost_exceeded` a moment ago now runs, and the audit entry
    shows the new revision. `make demo` scene 7 does exactly this and restores the file.

## The policy in this repository

The file below is included verbatim from
[`config/policy.yaml`](https://github.com/JMadejczyk/ai-security-gateway/blob/main/config/policy.yaml).

```yaml
--8<-- "config/policy.yaml"
```

The complete schema, with validation rules and cross-reference checks, is in the
[specification](spec.md#policy-file).
