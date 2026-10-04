"""``make opencode AS=anna|bartek``: opencode as a real coding agent, every model and tool call
through the gateway.

Each launch:

1. mints a fresh demo token for ``<AS>@demo`` on the ``opencode`` agent from the operator API
   (``POST /auth/demo-token``): a new gateway session (no taint, no risk) valid for one hour;
2. picks the model the selected upstream serves (``/v1/models``: ``qwen3:8b`` locally,
   ``deepseek-v4.1-flash`` remote);
3. starts the pinned opencode (``npx opencode-ai@1.18.34``) in a per-identity home under
   ``demo/opencode/.home/<AS>/`` (gitignored): ``HOME`` and every ``XDG_*`` directory point
   there, so nothing reads or writes the user's own opencode, Claude or npm configuration.

The token reaches opencode only through its environment (``ACL_OPENCODE_TOKEN``); the tracked
config (``demo/opencode/opencode.json``) references it as ``{env:ACL_OPENCODE_TOKEN}``. The
environment is built from an allowlist, so no provider key of the user's shell (OpenAI,
Anthropic, OpenRouter) reaches opencode either. Arguments after ``--`` go to opencode, e.g.
``run "How many customers do we have?"`` for one non-interactive turn.
"""

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import httpx

from demo.agent.acl_agent.client import as_object, pick_model

OPENCODE_PACKAGE: Final = "opencode-ai@1.18.34"
HERE: Final = Path(__file__).resolve().parent
HOMES: Final = HERE / ".home"
CONFIG: Final = HERE / "opencode.json"
IDENTITIES: Final = ("anna", "bartek")
AGENT: Final = "opencode"
# Passed through from the caller's environment; everything else (API keys, HOME, XDG) is not.
PASSTHROUGH: Final = ("PATH", "TERM", "COLORTERM", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "USER")


class LaunchError(RuntimeError):
    """The gateway refused or could not be reached; the message says what to check."""


def gateway_urls(env: Mapping[str, str]) -> tuple[str, str]:
    """(agent API, operator API) on the host ports compose publishes."""
    agent = env.get("ACL_AGENT_HOST_PORT", "8080")
    operator = env.get("ACL_OPERATOR_HOST_PORT", "9090")
    return f"http://127.0.0.1:{agent}", f"http://127.0.0.1:{operator}"


def mint_token(operator: str, who: str) -> dict[str, object]:
    """A fresh demo token (and session) for ``<who>@demo`` on the opencode agent."""
    try:
        response = httpx.post(
            f"{operator}/auth/demo-token",
            json={"sub": f"{who}@demo", "agent": AGENT},
            timeout=10,
        )
    except httpx.HTTPError as exc:
        msg = f"operator API at {operator} unreachable ({type(exc).__name__}): is the stack up?"
        raise LaunchError(msg) from None
    if response.status_code != httpx.codes.OK:
        code = as_object(as_object(response.json()).get("error")).get("code", response.status_code)
        msg = f"the gateway refused a token for {who}@demo on {AGENT}: {code}"
        raise LaunchError(msg)
    return as_object(response.json())


def prepare_home(who: str) -> tuple[Path, Path]:
    """The identity's isolated home and its workspace (its own git root, so opencode never
    takes this repository, or its instructions, as the project)."""
    home = HOMES / who
    workspace = home / "workspace"
    for directory in ("config", "data", "state", "cache"):
        (home / directory).mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    if not (workspace / ".git").exists():
        git = shutil.which("git") or "git"
        subprocess.run([git, "init", "-q", str(workspace)], check=True)  # noqa: S603 -- fixed argv
    return home, workspace


def opencode_env(
    caller: Mapping[str, str], home: Path, *, gateway: str, token: str, model: str
) -> dict[str, str]:
    env = {name: caller[name] for name in PASSTHROUGH if name in caller}
    env |= {
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / "config"),
        "XDG_DATA_HOME": str(home / "data"),
        "XDG_STATE_HOME": str(home / "state"),
        "XDG_CACHE_HOME": str(home / "cache"),
        "npm_config_cache": str(HOMES / "npm-cache"),
        "npm_config_update_notifier": "false",
        "OPENCODE_CONFIG": str(CONFIG),
        "OPENCODE_DISABLE_AUTOUPDATE": "1",
        "OPENCODE_DISABLE_LSP_DOWNLOAD": "1",
        "ACL_GATEWAY_URL": gateway,
        "ACL_OPENCODE_TOKEN": token,
        "ACL_OPENCODE_MODEL": model,
    }
    return env


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="make opencode", description=__doc__)
    parser.add_argument("--as", dest="who", choices=IDENTITIES, required=True)
    parser.add_argument("opencode_args", nargs=argparse.REMAINDER, help="passed to opencode")
    args = parser.parse_args(argv)
    passthrough = [a for a in args.opencode_args if a != "--"]
    agent_url, operator_url = gateway_urls(os.environ)
    try:
        issued = mint_token(operator_url, args.who)
    except LaunchError as exc:
        print(f"make opencode: {exc}", file=sys.stderr)
        return 2
    token = str(issued["access_token"])
    with httpx.Client(base_url=agent_url, timeout=10) as client:
        model = pick_model(client, token)
    if model is None:
        print("make opencode: /v1/models lists no model for this token", file=sys.stderr)
        return 2
    home, workspace = prepare_home(args.who)
    minutes = int(str(issued.get("expires_in", 3600))) // 60
    print(
        f"opencode as {args.who}@demo (agent {AGENT}), model {model}, gateway session "
        f"{issued.get('session_id')}; the token lasts {minutes} min: run `make opencode "
        f"AS={args.who}` again for a new one (and a new, clean session).",
        file=sys.stderr,
    )
    env = opencode_env(os.environ, home, gateway=agent_url, token=token, model=model)
    os.chdir(workspace)
    npx = shutil.which("npx")
    if npx is None:
        print("make opencode: npx (Node.js) is required", file=sys.stderr)
        return 2
    os.execve(npx, [npx, "-y", OPENCODE_PACKAGE, *passthrough], env)  # noqa: S606 -- pinned package


if __name__ == "__main__":
    sys.exit(main())
