"""``python -m gateway``: the agent and operator listeners in one process.

Bind addresses come from ``ACL_AGENT_HOST``/``ACL_AGENT_PORT`` and
``ACL_OPERATOR_HOST``/``ACL_OPERATOR_PORT`` (both default to 127.0.0.1). Compose binds the
operator listener to the gateway's address on the ``ops`` network only, so the agent on
``edge`` cannot mint tokens or reload the policy.
"""

import asyncio
import contextlib
import logging
import signal
import sys
from collections.abc import Generator

import uvicorn
from fastapi import FastAPI
from pydantic import ValidationError

from gateway.container import GatewayContainer
from gateway.feed.schema import FeedError
from gateway.injection.manifest import ModelVerificationError
from gateway.main import create_agent_app, create_operator_app
from gateway.policy.loader import PolicyLoadError
from gateway.settings import Settings

logger = logging.getLogger("gateway")


class _Listener(uvicorn.Server):
    """A uvicorn server that leaves signal handling to `serve`, which stops both listeners."""

    @contextlib.contextmanager
    def capture_signals(self) -> Generator[None]:
        yield


def build_servers(
    settings: Settings, container: GatewayContainer
) -> tuple[uvicorn.Server, uvicorn.Server]:
    """The agent and operator servers, configured but not started."""

    def server(app: FastAPI, host: str, port: int) -> uvicorn.Server:
        config = uvicorn.Config(
            app, host=host, port=port, log_level=settings.log_level, lifespan="on"
        )
        return _Listener(config)

    return (
        server(create_agent_app(container), settings.agent_host, settings.agent_port),
        server(create_operator_app(container), settings.operator_host, settings.operator_port),
    )


async def _listen(listener: uvicorn.Server) -> int:
    """Run one listener; its exit code (uvicorn exits the process when it cannot bind)."""
    try:
        await listener.serve()
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    return 0


async def serve(settings: Settings, container: GatewayContainer) -> int:
    """Run both listeners until a signal arrives or either one stops; returns the exit code."""
    servers = build_servers(settings, container)

    def stop() -> None:
        for listener in servers:
            listener.should_exit = True

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop)
    tasks = [asyncio.create_task(_listen(listener)) for listener in servers]
    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    stop()  # one listener ending (or failing to start) takes the other down with it
    codes = await asyncio.gather(*tasks)
    return max(codes)


def _settings_error(error: ValidationError) -> str:
    fields = ", ".join(
        f"ACL_{'.'.join(str(p) for p in item['loc']).upper()}: {item['msg']}"
        for item in error.errors(include_url=False, include_input=False)
    )
    return f"invalid settings: {fields}"


def main() -> int:
    try:
        settings = Settings()  # pyright: ignore[reportCallIssue] -- secrets come from ACL_* env vars
    except ValidationError as exc:
        print(_settings_error(exc), file=sys.stderr)
        return 2
    logging.basicConfig(
        level=settings.log_level.upper(), format="%(levelname)s %(name)s %(message)s"
    )
    try:
        container = GatewayContainer.from_settings(settings)
    except (PolicyLoadError, FeedError, ModelVerificationError, OSError, ValidationError) as exc:
        logger.critical("refusing to start: %s", exc)
        return 2
    return asyncio.run(serve(settings, container))


if __name__ == "__main__":
    sys.exit(main())
