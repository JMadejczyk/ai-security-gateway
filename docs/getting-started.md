# Run it

## Requirements

- Docker with Compose v2, `make`, and [uv](https://docs.astral.sh/uv/) (it fetches Python 3.12 itself).
- About 10 GB of free disk for the local model; 16 GB of RAM recommended.

## Two ways to run the models

The gateway, the controls and the demo are the same in both.

=== "A · Fully local (default)"

    `qwen3:8b` on Ollama in Docker. No accounts and no paid APIs; nothing leaves the machine.
    On a CPU-only laptop a model turn can take up to a minute.

=== "B · OpenRouter + DeepSeek"

    Model and AI-judge calls go to `deepseek/deepseek-v4.1-flash` on OpenRouter's
    **zero-data-retention** endpoints only, so turns take seconds. Needs an OpenRouter API key.
    Personal data is masked, and secrets and injections are blocked, *before* anything leaves the
    machine; the policy pins zero retention and a price ceiling the agent cannot override.
    Details: [OpenRouter mode](remote.md).

## 1 · Get the code and create `.env`

```bash
git clone https://github.com/JMadejczyk/ai-security-gateway.git
cd ai-security-gateway
python3 -c "import re,secrets;s=open('.env.example').read();open('.env','w').write(re.sub(r'=change-me\S*',lambda m:'='+secrets.token_urlsafe(32),s))"
```

That replaces every placeholder secret with a random one. For option B, also add your key:

```bash
echo "OPENROUTER_API_KEY=sk-or-..." >> .env
```

## 2 · One-time downloads

```bash
docker compose --profile init run --rm models-init   # injection classifier: both modes
docker compose --profile init run --rm ollama-init   # local model, ~5 GB: option A only
```

The gateway refuses to start with `prompt_injection` enabled and no verified classifier.

## 3 · Start the stack

=== "A · Local"

    ```bash
    make install
    make demo-up
    ```

=== "B · OpenRouter"

    ```bash
    make install
    make demo-up REMOTE=1
    ```

| What | Where |
| --- | --- |
| Agent API (models `/v1`, tools `/mcp`) | `http://127.0.0.1:8080` |
| Operator API | `http://127.0.0.1:9090` (`/healthz` shows `"llm_upstream": "local"` or `"remote"`) |
| Grafana | `http://127.0.0.1:3300`, user `admin`, password `ACL_GRAFANA_ADMIN_PASSWORD` from `.env` |

If a port is taken, set `ACL_OPERATOR_HOST_PORT`, `ACL_AGENT_HOST_PORT` or
`ACL_GRAFANA_HOST_PORT` in `.env`. From B back to A: `make demo-off REMOTE=1`, then `make demo-up`.

## 4 · See it work

```bash
make demo    # 7 narrated scenes; exits 1 on any deviation; works in both modes
make smoke   # mixed traffic, so the Grafana dashboards fill up
```

The scenes: identity down to the rows, an injection that taints the session, a night job held
for human approval, secrets and PII, budgets, and a live policy change. See the
[demo walkthrough](demo.md).

## 5 · Run the test suite

```bash
make test     # 3,034 tests + a check that all 15 controls have passing allow AND deny tests
make models   # optional: fetch the classifier so the real-model tests run too
```

No Docker and no API key needed. More in [Test suite](testing.md).

## 6 · Try your own prompts

```bash
TOKEN=$(curl -s -X POST 127.0.0.1:9090/auth/demo-token -H 'content-type: application/json' \
  -d '{"sub":"anna@demo"}' | python3 -c "import sys,json;print(json.load(sys.stdin)['access_token'])")
curl -s 127.0.0.1:8080/v1/chat/completions -H "authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"My AWS key is AKIAQ3EGRVW6XKZT4M7N, which region?"}]}'
# {"error":{"message":"blocked by policy: secret_detected", ...}}
```

The model is `deepseek-v4.1-flash` in option B and `qwen3:8b` in option A;
`GET /v1/models` with the token lists what is served. Demo identities: `anna@demo` (analyst)
and `bartek@demo` (intern). Any OpenAI-compatible client works against `/v1` with that token,
and `make opencode AS=anna` runs a real coding agent through the gateway
([guide](opencode.md)).

## 7 · Change the rules live

Edit `config/policy.yaml` (controls, thresholds, block vs redact, allowed models, budgets, the
OpenRouter settings) or `config/feeds/signatures.json` (the attack-signature feed). The gateway
hot-reloads, records the new revision in the audit log and marks it on the dashboards. An
invalid edit is rejected and the last valid policy stays in force. See [Policy file](policy.md).

## Why an LLM?

1. **It's the model the agents use.** The gateway is a proxy in front of it, so the demo needs
   a real model for opencode and DataBot to reason and call tools.
2. **It powers the AI judges**, the semantic half of the hybrid defense. Rules catch patterns
   (keys, PESEL, SQL cost, known attack signatures). A small LLM judges what depends on meaning:
   paraphrased prompt injections, hidden instructions in MCP tool descriptions, tool calls that
   don't fit the user's request (held for approval), and leaks in the model's answers (masked).

Cheap first: rules on every call, a local non-LLM classifier on inputs, the LLM only for
uncertain cases. A judge that fails or times out fails closed. Without the `judges:` section
in `config/policy.yaml` the AI checks turn off and the rules keep working. `make test` needs no
LLM.
