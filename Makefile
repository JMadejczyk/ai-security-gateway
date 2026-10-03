.DEFAULT_GOAL := help
REPORTS := reports

.PHONY: help install models lint fmt test report test-docker test-all perf up down grafana smoke dashboards

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
