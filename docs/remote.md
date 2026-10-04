# OpenRouter mode (DeepSeek)

The product runs fully local. For laptops without a GPU, the same stack can send model and AI-judge calls to **DeepSeek V4.1 Flash on OpenRouter**, zero-data-retention endpoints only: `make demo-up REMOTE=1`.

CPU-only Ollama runs qwen3:8b at a few tokens a second. For engineering, `make remote-up`
starts the stack with `compose.remote.yml`: the gateway sends `generate` and judge calls to
`upstreams.llm_remote` in `config/policy.yaml` (OpenRouter) instead of `upstreams.llm`. Put
`OPENROUTER_API_KEY=...` in `.env` first; it reaches the gateway container only, and the
gateway refuses to start without it (never a silent fall back to Ollama). `make remote-off`
returns to the local default, which is what the product ships.

- **Data.** Prompts and judge content leave the machine, after the pre controls: `pii` masks,
  `secrets` and `prompt_injection` block, before the upstream call. Every request carries the
  policy's `extra_body` (`provider: {zdr: true, data_collection: deny}`), which the agent
  cannot override; OpenRouter answers an error rather than route to an endpoint that retains
  data, and the gateway turns that into a generic 502 `upstream_error`. Router extensions in
  the agent's request (`models`, `provider`, `plugins`, ...) are dropped.
- **Models.** The remote upstream serves its own logical ids: `deepseek-v4.1-flash`
  (`deepseek/deepseek-v4.1-flash`, ~27 zero-data-retention endpoints). Grants, pricing,
  `model_allowlist` and audit use that id; `model_map` names the provider model it is sent as.
  `qwen3:8b` stays local-only: asking for it in remote mode (or for `deepseek-v4.1-flash` in
  local mode) is a 400 `model_not_mapped` that names what is served. `/v1/models` lists what
  the selected upstream serves, and the demo agent and smoke traffic pick their model from it,
  so `make demo` runs in both modes. Judges ask for `upstreams.llm_remote.judge_model` while
  remote is selected; remote mode with judges refuses to start without a mapped one. Remote
  judge verdicts can differ between runs of the same content, because different ZDR endpoints
  serve different runs, so they are less reproducible than the local judge's.
- **Visibility.** `/healthz` reports `llm_upstream` and its host, every LLM audit entry carries
  `"upstream": "remote"`, `acl_llm_upstream_info{upstream}` is 1 for the active one, and the
  gateway logs a warning at startup.
- **Budgets.** Remote calls are charged tokens at `llm_remote.pricing` and no GPU time; their
  deadline stays `limits.upstream_timeout_s`. `extra_body` also caps the endpoint price
  (`provider.max_price`) and `pricing` reserves at that ceiling, so a hold is never too small;
  the settlement charges the cost OpenRouter reports (`usage.cost`), never above the ceiling,
  and the ceiling when no usable figure comes back.
- **With the demo.** `make demo-up REMOTE=1` combines both overlays; `make remote-off` (or
  `make demo-off`) returns to the default.
