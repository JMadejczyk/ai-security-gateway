"""``python -m gateway``: settings from the environment, refusal to start, server wiring."""

import pytest
from fastapi import FastAPI
from gateway_testkit import INTERNAL_KEY, JWT_SECRET, ROOT_POLICY, make_settings
from starlette.routing import Route

from gateway import __main__ as entrypoint
from gateway.container import GatewayContainer
from gateway.main import create_agent_app, create_operator_app
from gateway.settings import Settings


@pytest.fixture
def env(monkeypatch):
    for name in (
        "ACL_AGENT_HOST",
        "ACL_AGENT_PORT",
        "ACL_OPERATOR_HOST",
        "ACL_OPERATOR_PORT",
        "ACL_DEMO_TOKENS",
        "ACL_AUDIT_PATH",
        "ACL_POLICY_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ACL_JWT_SECRET", JWT_SECRET)
    monkeypatch.setenv("ACL_INTERNAL_KEY", INTERNAL_KEY)
    return monkeypatch


def test_defaults_bind_loopback(env):
    settings = Settings()  # pyright: ignore[reportCallIssue]
    assert (settings.agent_host, settings.agent_port) == ("127.0.0.1", 8080)
    assert (settings.operator_host, settings.operator_port) == ("127.0.0.1", 9090)
    assert settings.demo_tokens is True
    assert settings.audit_path is None


def test_compose_environment_is_honoured(env):
    env.setenv("ACL_AGENT_HOST", "0.0.0.0")  # noqa: S104 -- the value compose sets for edge
    env.setenv("ACL_OPERATOR_HOST", "172.29.90.10")
    env.setenv("ACL_OPERATOR_PORT", "9191")
    env.setenv("ACL_DEMO_TOKENS", "0")
    settings = Settings()  # pyright: ignore[reportCallIssue]
    assert (settings.operator_host, settings.operator_port) == ("172.29.90.10", 9191)
    assert settings.demo_tokens is False


@pytest.mark.parametrize(
    ("name", "value"),
    [("ACL_JWT_SECRET", None), ("ACL_JWT_SECRET", "short"), ("ACL_INTERNAL_KEY", "")],
    ids=["missing-jwt", "short-jwt", "empty-internal"],
)
def test_refuses_to_start_without_strong_secrets(env, capsys, name, value):
    if value is None:
        env.delenv(name)
    else:
        env.setenv(name, value)
    assert entrypoint.main() == 2
    err = capsys.readouterr().err
    assert name in err
    assert JWT_SECRET not in err


def test_refuses_to_start_without_a_valid_policy(env, tmp_path):
    broken = tmp_path / "policy.yaml"
    broken.write_text("default: allow\n")
    env.setenv("ACL_POLICY_PATH", str(broken))
    assert entrypoint.main() == 2


def test_builds_two_listeners_from_settings():
    settings = make_settings(
        ROOT_POLICY,
        agent_host="0.0.0.0",  # noqa: S104 -- compose value, never bound in this test
        agent_port=18080,
        operator_host="172.29.90.10",
        operator_port=19090,
    )
    container = GatewayContainer.from_settings(settings, env={})
    agent, operator = entrypoint.build_servers(settings, container)
    assert isinstance(agent.config.app, FastAPI)
    assert isinstance(operator.config.app, FastAPI)
    assert (agent.config.host, agent.config.port) == ("0.0.0.0", 18080)  # noqa: S104
    assert (operator.config.host, operator.config.port) == ("172.29.90.10", 19090)
    agent_paths = {r.path for r in agent.config.app.routes if isinstance(r, Route)}
    operator_paths = {r.path for r in operator.config.app.routes if isinstance(r, Route)}
    assert {"/v1/chat/completions", "/v1/models", "/v1/session"} <= agent_paths
    assert {"/healthz", "/metrics", "/auth/demo-token", "/admin/reload"} <= operator_paths
    assert not agent_paths & {"/auth/demo-token", "/admin/reload", "/metrics"}


async def test_lifespan_starts_and_stops_shared_resources():
    settings = make_settings(ROOT_POLICY, policy_watch=True)
    container = GatewayContainer.from_settings(settings, env={})
    agent_app, operator_app = create_agent_app(container), create_operator_app(container)
    async with operator_app.router.lifespan_context(operator_app):
        async with agent_app.router.lifespan_context(agent_app):  # shares the started resources
            assert container.llm._client is not None
            assert container._watcher is not None
        assert container.llm._client is not None  # still in use by the operator app
    assert container.llm._client is None
    assert container._watcher is None
