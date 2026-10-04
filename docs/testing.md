# Test suite

`make test` runs about 2,990 tests in about 75 seconds on a fresh clone (3,034 once `make models` has fetched the classifier for the real-model tests), without Docker or any model, and fails unless every one of the 15 controls has a passing *allow* test and a passing *deny-side* test. The latest per-control table is on [Control coverage](reports/controls.md).

`make test` runs every test that needs no docker (`pytest -m "not docker"`) and fails unless
every control in the catalog has a passing allow test and a passing deny-side test.

| Layer | What it proves |
| --- | --- |
| `tests/unit/` | Each control alone, as case tables (input → verdict). Includes `egress`, the plugin and the analyzers. |
| `tests/policy/` | The role × agent × action × resource matrix, and default deny in every profile. |
| `tests/session_modes/` | Taint and risk remove actions in an interactive session. An autonomous session gets throttled or queued for approval instead. |
| `tests/approvals/` | The approval queue end to end: approve, deny, timeout, replay, altered arguments, self-approval. Also the kill switch. |
| `tests/identity/`, `tests/sessions/` | Forged, expired and wrong-audience tokens. Session binding, lifetimes and ended sessions. |
| `tests/llm/`, `tests/mcp/` | Both entry points through the real apps, with mocked LLMs and in-process MCP servers. |
| `tests/attacks/` | The curated attack corpus `corpus.yaml` (about 60 cases), described below. |
| `tests/budget/`, `tests/reload/` | Budget exhaustion and loop detection. Policy and feed changes mid-test. |
| `tests/injection/` | The real ONNX classifier against its corpus. |
| `tests/perf/` | Gateway overhead (`make perf`; a smoke version runs in `make test`). |
| `tests/bypass/`, `tests/e2e/` | Against the live compose stack (`docker` marker, `make test-docker`). |

`tests/attacks/` covers direct and indirect prompt injection, secret and PII leakage, tool
poisoning and rug pulls, SQL attacks, path traversal, SSRF, resource abuse, approval abuse and
token or session abuse. Each case also has benign look-alikes. The runner
`test_attack_corpus.py` checks each case's decision, reason code, audit verdicts, upstream
contact, taint and leaks. Each case points to the deep tests behind it.

Markers:

- `control(<id>, outcome)` declares what a test proves. `outcome` is one of `allow`, `deny`,
  `redact`, `require_approval` or `log_only`. Unknown ids, and outcomes the control does not
  support, fail at collection. The plugin is `tests/plugins/control_report.py`.
- `docker` needs `make up`.
- `redis` uses a throwaway `redis:7.2` container, or `ACL_TEST_REDIS_URL`. It skips without
  docker.
- `model` needs the pinned classifier in `models/cache` (`make models`). It skips without it.
- `perf` is selected only by `make perf`.

Reports land in `reports/`:

- `junit.xml`. Each claim is a `control` property.
- `report.html`, self-contained, with the per-control table embedded.
- `controls.md` and `controls.json`: per control, its catalog metadata, passing and total tests
  per outcome, and example test ids.

`make report` rebuilds `controls.md` and `controls.json` from the last `junit.xml`. A committed snapshot of the per-control table is in [`reports/controls.md`](reports/controls.md). A subset
run (`uv run pytest tests/mcp`) prints the coverage line but does not enforce it. Add
`--control-coverage` to enforce it, or `--control-report=DIR` to write the files.
`make test-docker` and `make test-all` add the live-stack layers.
