"""`make opencode`: a fresh token per launch, an isolated home, no secret in tracked files."""

import json
from pathlib import Path

import httpx
import pytest
import respx

from demo.opencode import launch

OPERATOR = "http://127.0.0.1:9090"


def test_the_environment_is_an_allowlist_and_everything_points_into_the_isolated_home(tmp_path):
    caller = {
        "PATH": "/usr/bin",
        "TERM": "xterm-256color",
        "HOME": "/Users/someone",
        "XDG_CONFIG_HOME": "/Users/someone/.config",
        "OPENROUTER_API_KEY": "user-secret-1",
        "OPENAI_API_KEY": "user-secret-2",
        "ANTHROPIC_API_KEY": "user-secret-3",
    }
    minted = "minted-" + "t" * 8  # stands in for the demo token
    env = launch.opencode_env(caller, tmp_path, gateway="http://gw", token=minted, model="m")
    assert not {"OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"} & set(env)
    assert "user-secret" not in json.dumps(env)
    assert env["HOME"] == str(tmp_path)
    for name in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME"):
        assert env[name].startswith(str(tmp_path)), name
    assert (env["ACL_OPENCODE_TOKEN"], env["ACL_OPENCODE_MODEL"]) == (minted, "m")
    assert env["ACL_GATEWAY_URL"] == "http://gw"
    assert env["OPENCODE_CONFIG"] == str(launch.CONFIG)
    assert (env["PATH"], env["TERM"]) == ("/usr/bin", "xterm-256color")


def test_the_tracked_config_holds_no_secret_and_routes_both_channels_through_the_gateway():
    text = launch.CONFIG.read_text()
    config = json.loads(text)
    acl = config["provider"]["acl"]
    assert acl["options"]["apiKey"] == "{env:ACL_OPENCODE_TOKEN}"
    assert acl["options"]["baseURL"] == "{env:ACL_GATEWAY_URL}/v1"
    assert config["enabled_providers"] == ["acl"]  # no other provider, no direct model access
    for server in ("sales_db", "web", "reports"):
        mcp = config["mcp"][server]
        assert mcp["url"] == f"{{env:ACL_GATEWAY_URL}}/mcp/{server}"
        assert mcp["headers"] == {"Authorization": "Bearer {env:ACL_OPENCODE_TOKEN}"}
    # opencode's own tools run on the host, outside the gateway: switched off.
    for tool in ("bash", "edit", "write", "read", "webfetch", "task"):
        assert config["tools"][tool] is False, tool
    assert config["default_agent"] == "databot"
    # opencode's title generator would be the session's first model call, and the gateway
    # records the first user message it forwards as the intent judge's goal.
    assert config["agent"]["title"] == {"disable": True}
    assert (config["autoupdate"], config["share"]) == (False, "disabled")
    model = acl["models"]["{env:ACL_OPENCODE_MODEL}"]
    assert model["options"] == {"reasoningEffort": "none"}  # sent as reasoning_effort: none


def test_each_launch_mints_a_fresh_token_for_the_opencode_agent():
    issued = {"access_token": "t", "session_id": "s-1", "expires_in": 3600}
    with respx.mock(base_url=OPERATOR) as router:
        route = router.post("/auth/demo-token").mock(return_value=httpx.Response(200, json=issued))
        assert launch.mint_token(OPERATOR, "anna") == issued
    assert json.loads(route.calls.last.request.content) == {"sub": "anna@demo", "agent": "opencode"}


def test_a_refused_token_names_the_reason():
    refusal = {"error": {"code": "principal_not_allowed", "message": "token refused"}}
    with respx.mock(base_url=OPERATOR) as router:
        router.post("/auth/demo-token").mock(return_value=httpx.Response(403, json=refusal))
        with pytest.raises(launch.LaunchError, match="principal_not_allowed"):
            launch.mint_token(OPERATOR, "anna")


def test_the_workspace_is_its_own_git_root(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "HOMES", tmp_path)
    home, workspace = launch.prepare_home("bartek")
    assert home == tmp_path / "bartek"
    assert (workspace / ".git").is_dir()  # opencode never takes this repository as its project
    assert all((home / d).is_dir() for d in ("config", "data", "state", "cache"))


def test_the_isolated_homes_are_gitignored():
    gitignore = (Path(launch.__file__).resolve().parents[2] / ".gitignore").read_text()
    assert "demo/opencode/.home/" in gitignore.splitlines()
