"""The remote LLM overlay (compose.remote.yml): opt-in, its key reaches the gateway only.

Static: renders ``docker compose config`` with placeholder secrets and never the developer's
``.env`` (needs only the docker CLI). The agent's isolation and the gateway's way out are the
same with and without the overlay: it opens no network.
"""

from __future__ import annotations

import pytest
from compose_support import REPO_ROOT, ComposeConfig, ComposeConfigError, find_docker

REMOTE_OVERLAY = REPO_ROOT / "compose.remote.yml"
KEY_ENV = "OPENROUTER_API_KEY"
PLACEHOLDER = {KEY_ENV: "placeholder-openrouter-key"}


@pytest.fixture(scope="module")
def docker() -> str:
    found = find_docker()
    if found is None:
        pytest.skip("docker CLI not installed")
    return found


@pytest.fixture(scope="module")
def base(docker: str) -> ComposeConfig:
    return ComposeConfig.render(docker, extra_env=PLACEHOLDER)


@pytest.fixture(scope="module")
def remote(docker: str) -> ComposeConfig:
    return ComposeConfig.render(docker, overlays=(REMOTE_OVERLAY,), extra_env=PLACEHOLDER)


def holders(config: ComposeConfig, name: str) -> set[str]:
    return {s for s in config.services if name in config.environment_of(s)}


def test_the_default_stack_never_carries_the_key_or_selects_remote(base: ComposeConfig) -> None:
    assert holders(base, KEY_ENV) == set()
    assert "ACL_LLM_UPSTREAM" not in base.environment_of("gateway")  # default: local


def test_the_overlay_gives_the_key_to_the_gateway_only(remote: ComposeConfig) -> None:
    assert holders(remote, KEY_ENV) == {"gateway"}
    gateway = remote.environment_of("gateway")
    assert gateway["ACL_LLM_UPSTREAM"] == "remote"
    assert gateway[KEY_ENV] == PLACEHOLDER[KEY_ENV]
    assert not remote.service("agent").get("env_file")


def test_the_overlay_refuses_to_render_without_the_key(docker: str) -> None:
    with pytest.raises(ComposeConfigError, match=KEY_ENV):
        ComposeConfig.render(docker, overlays=(REMOTE_OVERLAY,), drop_env=(KEY_ENV,))


@pytest.mark.parametrize("variant", ["base", "remote"])
def test_no_network_changes_the_gateway_leaves_through_ops_the_agent_stays_on_edge(
    request: pytest.FixtureRequest, variant: str
) -> None:
    config: ComposeConfig = request.getfixturevalue(variant)
    gateway_out = {
        n for n in config.networks_of("gateway") if not config.networks[n].get("internal")
    }
    assert gateway_out == {"ops"}  # its only route to the internet, in both stacks
    assert config.networks_of("agent") == {"edge"}
    assert config.networks["edge"].get("internal") is True
