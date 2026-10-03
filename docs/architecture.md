# Architecture

The AI Control Layer is one gateway process with two entry points, an OpenAI-compatible LLM
proxy (`/v1`) and an MCP proxy (`/mcp`), in front of a shared policy core. Every model call and
every tool call goes through the same pipeline and reads and writes the same session state. A
detection on one channel therefore changes what the agent may do on the other. `SPEC.md` is the
source of truth. This page explains the shape of the running system in three pictures.

Rendered copies of the diagrams (SVG, and PNG at 2x, each in a light and a `-dark` variant) live
in [`img/architecture/`](img/architecture/). `make diagrams` re-renders them from this file.

## System context and network topology

The agent can reach exactly one thing: the gateway's agent listener on the internal `edge`
network. The upstreams sit on internal networks that they share only with the gateway, grouped
by trust, so the agent reaches none of them and a compromised fetch server cannot reach
Postgres, Ollama or Redis. The gateway joins `edge`, `ops`, `llm_backend`, `mcp_backend`,
`mcp_untrusted` and `state`; each arrow from it enters the network its target sits on. `ops`
is not internal because it carries the two published ports, the operator listener and Grafana,
both bound to `127.0.0.1`. Apart from `ops`, only mcp-fetch (through `fetch_egress`) and the
one-off `bootstrap` jobs have an internet route. The demo overlay `demo/compose.demo.yml` adds
`demo-web`, which serves the page with the hidden injection, on an internal `demo_web` network
it shares only with mcp-fetch (dashed). The sources of truth are `docker-compose.yml` and that
overlay, and `tests/unit/test_architecture_doc.py` fails if one of their services or networks
is missing from this diagram.

<!-- topology diagram: every compose service and network must appear in it (tests/unit/test_architecture_doc.py) -->
```mermaid
flowchart LR
    host(["operator / developer<br/>host 127.0.0.1"])

    subgraph edge["edge · internal"]
        agent["agent"]
    end

    gateway["<b>gateway</b> · one process<br/>:8080 agent listener · /v1 /mcp<br/>policy core + pipeline · credentials<br/>:9090 operator listener · /auth /admin /metrics"]

    subgraph llm_backend["llm_backend · internal"]
        ollama["ollama<br/>agent model + LLM judge"]
    end

    subgraph mcp_backend["mcp_backend · internal"]
        direction TB
        mcppg["mcp-postgres<br/>trust: internal"]
        postgres[("postgres<br/>FORCE RLS")]
        mcpfiles["mcp-files<br/>trust: internal"]
        mcppg -- "acl.set_principal + SELECT" --> postgres
    end

    subgraph mcp_untrusted["mcp_untrusted · internal"]
        mcpfetch["mcp-fetch<br/>trust: untrusted"]
    end

    subgraph fetch_egress["fetch_egress · external · mcp-fetch only"]
        net1(["internet"])
    end

    subgraph demo_web["demo_web · internal · demo overlay only"]
        demoweb["demo-web<br/>injected page"]
    end

    subgraph state["state · internal"]
        redis[("redis<br/>sessions · budgets<br/>approvals · kill switch")]
    end

    subgraph ops["ops · external (published ports)"]
        direction TB
        prometheus["prometheus"]
        alloy["alloy"]
        loki["loki"]
        grafana["grafana<br/>127.0.0.1:3300"]
        alloy --> loki
        grafana --> prometheus
        grafana --> loki
    end

    subgraph bootstrap["bootstrap · external · profile init, one-off"]
        direction TB
        ollamainit["ollama-init<br/>→ ollama_models volume"]
        modelsinit["models-init<br/>→ injection_model volume"]
        net2(["internet"])
        ollamainit -- "pull model" --> net2
        modelsinit -- "pinned classifier" --> net2
    end

    agent -- "Bearer JWT<br/>:8080" --> gateway
    host -- ":9090" --> gateway
    host -. ":8080" .-> gateway
    host -- ":3300" --> grafana
    gateway --> ollama
    gateway -- "X-ACL-Principal" --> mcppg
    gateway --> mcpfiles
    gateway --> mcpfetch
    mcpfetch --> net1
    mcpfetch -.-> demoweb
    gateway --> redis
    prometheus -- "scrape :9090" --> gateway
    gateway -. "audit_log volume" .-> alloy

    classDef gw fill:#dbe7ff,stroke:#2f5fb3,color:#0b1f44,stroke-width:2px
    classDef trusted fill:#e3f1e6,stroke:#3b7d4a,color:#10301a
    classDef untrusted fill:#fde2e1,stroke:#b3392f,color:#4a0f0b,stroke-width:2px
    classDef obs fill:#efe9f7,stroke:#6b51a0,color:#25173f
    classDef ext fill:#fff4d6,stroke:#a77b0c,color:#3d2c00
    classDef actor fill:#f1f1f1,stroke:#555,color:#111
    class gateway gw
    class ollama,mcppg,postgres,mcpfiles,redis trusted
    class mcpfetch,demoweb untrusted
    class prometheus,alloy,loki,grafana obs
    class net1,net2,ollamainit,modelsinit ext
    class agent,host actor
```

The network details (which service holds which secret, why the operator listener binds the
`ops` address instead of `0.0.0.0`, why `mcp_untrusted` is internal) are in
[`demo/README.md`](../demo/README.md#network-topology). `tests/bypass/` checks the topology from
inside the agent and mcp-fetch containers.

## Request pipeline

Both channels run the same steps in `gateway/pipeline.py`; the channel only decides the adapter
and the upstream. Every step reads the one policy snapshot taken when the call was admitted, and
the session stays locked until the outcome is persisted, so the next call already sees the new
risk and taint. Once the token is verified, a refusal at any step is still persisted and audited, with the
risk its verdicts add. The classifier and the LLM judge sit inside the semantic controls, the
approval queue behind the merge, and the kill switch is checked at admission, right before
dispatch and again before the result is released.

```mermaid
flowchart LR
    subgraph admit["1 · Admit"]
        direction TB
        req["agent call<br/>LLM /v1/chat/completions<br/>MCP tools/call"]
        authn["authn<br/>JWT · sub + act · mode"]
        lock["lock session<br/>+ policy snapshot"]
        adapt["adapter → Interactions<br/>(action, resource)"]
        req --> authn --> lock --> adapt
    end

    subgraph decide["2 · Decide"]
        direction TB
        authz["base authz<br/>U ∩ A ∩ T, deny, blocklist<br/>+ session restrictions"]
        admitchk["kill switch<br/>approval id · intent flags"]
        det["pre · deterministic<br/>tool_pinning · egress · secrets · pii<br/>signatures · model_allowlist · loop_detect"]
        sem["pre · semantic<br/>prompt_injection<br/>tool_poisoning"]
        authz --> admitchk --> det --> sem
    end

    subgraph oblig["3 · Merge and obligations"]
        direction TB
        merge["merge verdicts<br/>block > approval > redact > allow"]
        redact["apply redactions<br/>and rewrites"]
        seal["sql_guard seals final SQL<br/>EXPLAIN cost · forced LIMIT"]
        throttle["re-authorize rewrite<br/>throttle cap"]
        budget["budget<br/>reserve"]
        merge --> redact --> seal --> throttle --> budget
    end

    subgraph run["4 · Execute and record"]
        direction TB
        dispatch["kill switch<br/>consume approval"]
        upstream["upstream, once<br/>Ollama · MCP server"]
        post["post controls<br/>secrets · pii · signatures · prompt_injection<br/>model_allowlist · intent_judge · output_policy"]
        persist["persist<br/>risk · taint · timers"]
        audit["audit JSONL<br/>+ metrics"]
        result["result<br/>to agent"]
        dispatch --> upstream --> post --> persist --> audit --> result
    end

    admit --> decide --> oblig --> run

    classifier[["injection classifier<br/>ONNX, in-process"]]
    judge[["LLM judge<br/>JudgeClient → Ollama"]]
    queue[("approval queue<br/>Redis")]
    kill[("kill switch<br/>Redis")]
    operator(["operator<br/>/admin on :9090"])

    decide <-.-> classifier
    classifier -. "uncertain band" .-> judge
    run <-.-> judge
    oblig -- "require_approval:<br/>hold, return approval_id" --> queue
    operator -- "approve · deny" --> queue
    operator -- "kill · unkill" --> kill
    kill -.-> decide
    kill -.-> run
    queue -. "retry with approval_id" .-> decide

    classDef step fill:#dbe7ff,stroke:#2f5fb3,color:#0b1f44
    classDef semantic fill:#efe9f7,stroke:#6b51a0,color:#25173f
    classDef store fill:#e3f1e6,stroke:#3b7d4a,color:#10301a
    classDef actor fill:#f1f1f1,stroke:#555,color:#111
    class req,authn,lock,adapt,authz,admitchk,det,merge,redact,seal,throttle,budget,dispatch,upstream,persist,audit,result step
    class sem,post,classifier,judge semantic
    class queue,kill store
    class operator actor
```

The two channels differ only at the edges of this flow:

- **LLM.** The adapter yields one `generate:model:<name>` interaction. Post controls check the
  model that actually answered, and `intent_judge` compares each `tool_call` with the user's
  goal. It never holds the response: it flags the call in the session, and the matching MCP
  `tools/call` then needs approval (the intent-flag check in step 2). With `stream: true` the
  upstream call is still buffered, and SSE is re-emitted after the post controls.
- **MCP.** The adapter maps the tool through the operator-owned mapping in `policy.yaml`. An SQL
  query yields one interaction per table, and `sql_guard` gets its plan cost from the gateway-only
  `explain` tool on mcp-postgres. A result from a `trust: untrusted` server, such as the web
  fetch, taints the session even when it is blocked. The taint is persisted before the result
  is released.

## The thesis: a detection reshapes permissions

Base permissions come only from the policy: the principal's grants, the agent's grants and the
task scope, intersected per concrete `(action, resource)`. Nothing at runtime widens them, and
an approval never grants anything outside them. Detections feed a session state instead. Risk
decays with a half-life, and taint stays until the session ends. The session's mode then
decides what that state does: an interactive session loses actions, while an autonomous agent
keeps its grants but is throttled and sent to human approval. Demo scenes 2 and 3 run the same
injected page through both modes.

```mermaid
flowchart LR
    subgraph detect["Detections"]
        direction LR
        d1["prompt_injection hit<br/>risk +0.6, taint"]
        d2["untrusted tool result<br/>web fetch: taint"]
        d3["signatures +0.4 · secrets +0.3<br/>egress +0.3 · pii +0.1"]
    end

    subgraph base["Base grants · policy only, never widened"]
        direction TB
        T["T · task scope<br/>token scope claim"]
        A["A · agent's allow<br/>databot · nightly_etl"]
        U["U · principal's roles<br/>anna@demo: analyst"]
        I{{"U ∩ A ∩ T<br/>minus deny, blocklist"}}
        T --> I
        A --> I
        U --> I
    end

    S[("session state · Redis<br/>risk in [0, 1], half-life 600 s<br/>taint sticky until session end")]
    E["<b>effective permissions</b><br/>= base − restrictions<br/>+ obligations"]

    subgraph interactive["interactive session · narrow"]
        direction TB
        IA["taint: write, delete, egress removed<br/>risk > 0.5: write needs approval, 5 min cooldown<br/>risk > 0.8: tool calls frozen 5 min"]
        sc2["<b>Scene 2</b> · databot for anna<br/>write report: allowed<br/>fetch injected page: tainted<br/>same write: blocked"]
        IA --> sc2
    end

    subgraph autonomous["autonomous agent · hold, don't revoke"]
        direction TB
        AU["taint: write, delete, egress need approval<br/>risk > 0.5: throttle 1 per 10 s + backoff, alert<br/>risk > 0.8: all but read, generate need approval"]
        sc3["<b>Scene 3</b> · nightly_etl<br/>same injected page: tainted<br/>same write: held for approval<br/>agent throttled, not stopped<br/>olga approves: runs exactly once"]
        AU --> sc3
    end

    detect --> S
    base --> E
    S -- "risk_rules<br/>by mode" --> E
    E --> interactive
    E --> autonomous

    classDef grant fill:#e3f1e6,stroke:#3b7d4a,color:#10301a
    classDef det fill:#fde2e1,stroke:#b3392f,color:#4a0f0b
    classDef st fill:#fff4d6,stroke:#a77b0c,color:#3d2c00
    classDef eff fill:#dbe7ff,stroke:#2f5fb3,color:#0b1f44,stroke-width:2px
    classDef scene fill:#f1f1f1,stroke:#555,color:#111
    class U,A,T,I grant
    class d1,d2,d3 det
    class S st
    class E,IA,AU eff
    class sc2,sc3 scene
```

The rules come from `risk_rules` in `config/policy.yaml`, one list per session mode, and do not
depend on the strictness profile. Timers never extend themselves. A cooldown starts when a call
is denied above the threshold, a freeze starts when risk first crosses 0.8, and either one
applies again only if risk is still above its threshold when it ends. Only a human can disable
an agent permanently, with the kill switch. The audit entry of every call records its
`effective_scope`, `risk` and `taint`, so Grafana's Session trace dashboard shows these
transitions as they happen.
