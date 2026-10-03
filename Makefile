.DEFAULT_GOAL := help
REPORTS := reports

.PHONY: help install models lint fmt test test-docker test-all up down

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

test: ## Unit/policy/session/reload suites (no docker); JUnit + HTML in reports/
	@mkdir -p $(REPORTS)
	uv run pytest -m "not docker" --junitxml=$(REPORTS)/junit.xml \
		--html=$(REPORTS)/report.html --self-contained-html

test-docker: ## Tests that need the live compose stack (`make up` first)
	@mkdir -p $(REPORTS)
	uv run pytest -m docker --junitxml=$(REPORTS)/junit-docker.xml

test-all: ## Every test, including docker-marked ones
	@mkdir -p $(REPORTS)
	uv run pytest --junitxml=$(REPORTS)/junit-all.xml

up: ## Build and start the compose stack
	docker compose up -d --build

down: ## Stop the compose stack
	docker compose down
