.DEFAULT_GOAL := help
REPORTS := reports

.PHONY: help install models lint fmt test report test-docker test-all perf up down grafana smoke dashboards \
	demo demo-up demo-off remote-up remote-off opencode diagrams record

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install: ## Create .venv from uv.lock (dev tools included)
	uv sync --frozen

models: ## Fetch the pinned injection classifier into models/cache (real-model tests, local runs)
	uv run python -m gateway.injection.fetch --dest models/cache

lint: ## ruff check + ruff format --check + pyright (strict for gateway/)
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright

fmt: ## Apply ruff fixes and formatting
	uv run ruff check --fix .
	uv run ruff format .

test: ## Full suite without docker + control coverage check; junit.xml, report.html, controls.md in reports/
	@mkdir -p $(REPORTS)
	uv run pytest -m "not docker" --control-coverage --control-report=$(REPORTS) \
		--junitxml=$(REPORTS)/junit.xml --html=$(REPORTS)/report.html --self-contained-html

report: ## Rebuild reports/controls.md + controls.json from the last run's reports/junit.xml
	uv run python tests/plugins/control_report.py --junit $(REPORTS)/junit.xml --out $(REPORTS)

test-docker: ## Tests that need the live compose stack (`make up` first)
	@mkdir -p $(REPORTS)
	uv run pytest -m docker --junitxml=$(REPORTS)/junit-docker.xml

test-all: ## Every test, including docker-marked ones
	@mkdir -p $(REPORTS)
	uv run pytest --junitxml=$(REPORTS)/junit-all.xml

perf: ## Gateway overhead with mocked upstreams -> reports/perf.json + perf.md (ACL_PERF_ITERATIONS=200)
	@mkdir -p $(REPORTS)
	uv run pytest tests/perf -m perf -q

# A host port: the environment, else .env (compose reads it too), else the compose default.
port = $(or $($(1)),$(shell sed -n 's/^$(1)=//p' .env 2>/dev/null | tail -n 1),$(2))
GRAFANA_PORT = $(call port,ACL_GRAFANA_HOST_PORT,3300)
AGENT_PORT = $(call port,ACL_AGENT_HOST_PORT,8080)
OPERATOR_PORT = $(call port,ACL_OPERATOR_HOST_PORT,9090)

up: ## Build and start the compose stack (gateway, upstreams, Prometheus, Loki, Alloy, Grafana); waits until healthy
	docker compose up -d --build --wait
	@$(MAKE) --no-print-directory grafana

down: ## Stop the compose stack (volumes kept; `docker compose down -v` drops them)
	docker compose down

grafana: ## Print the Grafana URL (user admin, password ACL_GRAFANA_ADMIN_PASSWORD from .env)
	@echo "Grafana: http://127.0.0.1:$(GRAFANA_PORT)  (admin / ACL_GRAFANA_ADMIN_PASSWORD)"

smoke: ## Demo traffic through the running stack (allow, block, redact, approval, taint, reload)
	ACL_AGENT_HOST_PORT=$(AGENT_PORT) ACL_OPERATOR_HOST_PORT=$(OPERATOR_PORT) \
		uv run python -m observability.smoke_traffic $(SMOKE_ARGS)

dashboards: ## Regenerate grafana/dashboards/*.json from grafana/build_dashboards.py
	uv run python -m grafana.build_dashboards

# Mermaid CLI, pinned. Its first run downloads a headless Chromium for puppeteer.
MMDC := npx -y @mermaid-js/mermaid-cli@12.0.0 -q
ARCH_IMG := docs/img/architecture

diagrams: ## Render docs/architecture.md's diagrams to docs/img/architecture/ (SVG + 2x PNG, light and -dark; needs Node)
	@tmp=$$(mktemp -d) && trap 'rm -rf "$$tmp"' EXIT && \
	for variant in light dark; do \
		if [ $$variant = dark ]; then bg=transparent; suffix=-dark; else bg=white; suffix=; fi; \
		$(MMDC) -i docs/architecture.md -o $$tmp/d.svg -c $(ARCH_IMG)/config-$$variant.json -b $$bg && \
		$(MMDC) -i docs/architecture.md -o $$tmp/d.png -c $(ARCH_IMG)/config-$$variant.json -b $$bg -s 2 --size 1400 && \
		n=1 && for name in topology pipeline thesis; do \
			mv $$tmp/d-$$n.svg $(ARCH_IMG)/$$name$$suffix.svg && mv $$tmp/d-$$n.png $(ARCH_IMG)/$$name$$suffix.png; \
			n=$$((n + 1)); \
		done || exit 1; \
	done
	@ls $(ARCH_IMG)

# REMOTE=1 adds compose.remote.yml (OpenRouter instead of Ollama) to demo-up / demo-off.
REMOTE ?= 0
REMOTE_OVERLAY := $(if $(filter 1,$(REMOTE)),-f compose.remote.yml,)
DEMO_COMPOSE := docker compose -f docker-compose.yml -f demo/compose.demo.yml $(REMOTE_OVERLAY)

demo-up: ## Start the stack with the demo overlay (demo-web serves the injection page); REMOTE=1 adds the remote LLM overlay
	$(DEMO_COMPOSE) up -d --build --wait
	@$(MAKE) --no-print-directory grafana

demo: ## Run the 7-scene demo against the running stack (DEMO_ARGS="--pause" or "--scene 3"); exit 1 on any deviation
	ACL_OPERATOR_HOST_PORT=$(OPERATOR_PORT) ACL_GRAFANA_HOST_PORT=$(GRAFANA_PORT) \
		uv run python -m demo.run_demo $(DEMO_ARGS)

demo-off: ## Back to the default stack: gateway and mcp-fetch without the demo allowances, demo-web removed (REMOTE=1 keeps remote)
	docker compose -f docker-compose.yml $(REMOTE_OVERLAY) up -d --wait --remove-orphans

opencode: ## opencode (a real coding agent) through the gateway: AS=anna|bartek, ARGS='run "prompt"' for one turn
	@test -n "$(AS)" || { echo "usage: make opencode AS=anna|bartek [ARGS='run \"...\"']"; exit 2; }
	ACL_AGENT_HOST_PORT=$(AGENT_PORT) ACL_OPERATOR_HOST_PORT=$(OPERATOR_PORT) \
		uv run python -m demo.opencode.launch --as $(AS) -- $(ARGS)

REMOTE_COMPOSE := docker compose -f docker-compose.yml -f compose.remote.yml

remote-up: ## Engineering: LLM calls go to OpenRouter (ZDR only) instead of Ollama; needs OPENROUTER_API_KEY in .env
	$(REMOTE_COMPOSE) up -d --build --wait
	@curl -fsS http://127.0.0.1:$(OPERATOR_PORT)/healthz; echo

remote-off: ## Back to the default stack: the local Ollama upstream
	docker compose up -d --wait --remove-orphans

record: ## Stage one beat of the pitch video (docs/video.md): make record SCENE=1..6 [RECORD_ARGS="--pace 3 --open"]
	@test -n "$(SCENE)" || { echo "usage: make record SCENE=1..6"; exit 2; }
	ACL_OPERATOR_HOST_PORT=$(OPERATOR_PORT) ACL_GRAFANA_HOST_PORT=$(GRAFANA_PORT) \
		uv run python -m demo.record --scene $(SCENE) $(RECORD_ARGS)
