# opencode through the gateway

[opencode](https://opencode.ai) is a real coding agent. For the demo video it plays DataBot:
its model calls and its tools both go through the gateway, so the same policy, controls,
session risk and audit apply to it as to the demo agent. `docs/video.md` has the beats.

## Run it

```sh
make demo-up REMOTE=1                 # the stack, demo overlay (demo-web) and the remote LLM
make opencode AS=anna                 # the TUI, as anna@demo
make opencode AS=bartek ARGS='run "How many customers do we have?"'   # one turn, no TUI
```

`make opencode` (`demo/opencode/launch.py`):

1. Mints a fresh token for `<AS>@demo` on the `opencode` agent from the operator API
   (`POST /auth/demo-token`). Every launch is therefore a **new gateway session**: no taint, no
   risk, fresh session budgets. The token lives one hour (the issuer's maximum). A session
   that runs longer starts failing with `401 token_expired`; quit and run `make opencode` again
   (that is also how to reset a tainted session between takes).
2. Picks the model the selected upstream serves, from `/v1/models`: `deepseek-v4.1-flash` in
   remote mode, `qwen3:8b` on local Ollama.
3. Starts the pinned opencode (`npx -y opencode-ai@1.18.34`) in `demo/opencode/.home/<AS>/`
   (gitignored), with `HOME` and every `XDG_*` directory pointing there and the npm cache in
   `demo/opencode/.home/npm-cache/`. Nothing reads or writes `~/.config/opencode`, `~/.claude`
   or `~/.npm`. (opencode reads skills from `~/.claude/skills` when `HOME` is the real one;
   that is why `HOME` is isolated too.) The workspace is its own git repository, so opencode
   never takes this repo, or its instructions, as the project.

The token reaches opencode only through its environment (`ACL_OPENCODE_TOKEN`). The tracked
config, [`demo/opencode/opencode.json`](../demo/opencode/opencode.json), refers to it as
`{env:ACL_OPENCODE_TOKEN}` and holds no secret. The environment is built from an allowlist
(`PATH`, `TERM`, locale), so no provider key from your shell (OpenAI, Anthropic, OpenRouter)
reaches opencode: it can only talk to models through the gateway. Delete
`demo/opencode/.home/<AS>/` to clear opencode's local session history.

When scripting `run`, give it no stdin (`</dev/null`): `opencode run` reads piped stdin as
part of the message and waits for it to close.

**Multi-turn beats need one opencode process.** The TUI is one; for a script, start
`opencode serve` once and send each turn with `opencode run --attach <url> --continue`. A
fresh `opencode run` process per turn lists the MCP tools again, and after a taint the
gateway's filtered `tools/list` no longer offers `reports_write_report`: the model then says it
has no report tool instead of showing the gateway's `action_removed_by_session_risk`. Both are
the gateway working; only the long-lived process shows the refusal on screen.

## What goes through the gateway

| opencode | Gateway | Governed by |
| --- | --- | --- |
| Model calls (`provider.acl`, `@ai-sdk/openai-compatible`) | `/v1/chat/completions` with the agent token | authz, pii, secrets, signatures, model_allowlist, loop_detect, prompt_injection, budgets, intent_judge, output_policy; audited with `actor: opencode` |
| MCP `sales_db`, `web`, `reports` (`type: remote`, `Authorization: Bearer <token>`) | `/mcp/sales_db`, `/mcp/web`, `/mcp/reports` | authz on each `tools/call`, sql_guard and RLS, egress, tool_pinning, prompt_injection on results, session risk and taint |
| Its title generator (one short call per session, same model) | `/v1/chat/completions` | the same as any model call |

opencode sends what an AI SDK client sends: `stream: true` with
`stream_options.include_usage`, `tool_choice: auto`, `max_tokens` (the model's output limit in
the config, 2048), its own `x-opencode-*` headers (never forwarded upstream) and parallel tool
calls. `reasoning_effort: "none"` comes from the model's `options.reasoningEffort` in the
config, so neither DeepSeek nor qwen3 spends the turn thinking; the gateway does not need a
per-agent default. The gateway calls the upstream without streaming, runs the post controls
on the whole answer and re-emits it as `chat.completion.chunk` events the AI SDK parses (one
`tool_calls` delta per call with its index and id, `finish_reason: tool_calls`, then usage).
`tests/llm/test_opencode_requests.py` pins that shape.

The MCP tools appear to the model as `sales_db_query`, `web_fetch` and
`reports_write_report`. The intent judge's flags are matched under those prefixed names too
(`<server>_<tool>`, and Claude Code's `mcp__<server>__<tool>`), so a flagged `web_fetch` still
holds the `fetch` call on `web`.

opencode's title generator is disabled in the config (`agent.title`). It would be the
session's first model call, and the gateway records the first user message it forwards as the
intent judge's goal: the goal became "Generate a title for this conversation:" and every tool
call was judged misaligned.

## What does not

**opencode's built-in tools run on your machine, outside the gateway.** `bash`, file
read/edit/write, `glob`/`grep`, `webfetch`, `task` and the rest execute locally; the gateway
never sees them, so none of its controls apply (a `webfetch` would bypass `egress`; `bash`
could do anything your user can). The launcher's config switches them all off (`tools` in
`opencode.json`), which leaves the model with the three gateway tools only. Turn one back on
and that tool is ungoverned. The SPEC's in-process SDK extension point is how such tools
would be brought under the gateway; it is out of scope here.

## The DataBot prompt

opencode's stock system prompt (about 9.7 k characters with the built-in tools off, 19 k with
them on) is replaced by a short agent prompt,
[`demo/opencode/databot-prompt.md`](../demo/opencode/databot-prompt.md) (`agent.databot` in the
config). The stock prompt was blocked live: the injection classifier scored two of its
windows 0.91 and 0.82, and the judge confirmed them as instructions to an AI, which a system
prompt is by design. The DataBot prompt scores 0.00 and keeps each request small (about
2 k characters with the three tool definitions), which also keeps opencode's long sessions
well inside the per-user token budget.

## Identities

The `opencode` agent in `config/policy.yaml` is interactive, has databot's grants with both
models named, and may act for `anna@demo` and `bartek@demo` only (`principals`). Anna and
Bartek see different rows for the same query (RLS on `sub`). There is no autonomous opencode:
the night-job beat (`svc:nightly_etl`, approval queue) is shown from the demo runner.
