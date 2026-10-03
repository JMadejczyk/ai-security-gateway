"""Environment-driven settings shared by the demo servers."""

from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict, Field


class MissingSettingError(RuntimeError):
    """A required environment variable is not set."""

    def __init__(self, name: str) -> None:
        super().__init__(f"{name} must be set")


def required_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise MissingSettingError(name)
    return value


class ListenSettings(BaseModel):
    """Where the streamable HTTP endpoint listens. Compose sets MCP_HOST=0.0.0.0."""

    model_config = ConfigDict(frozen=True)

    host: str = "127.0.0.1"
    port: int = Field(default=8000, gt=0, lt=65536)
    path: str = "/mcp"

    @classmethod
    def from_env(cls) -> ListenSettings:
        return cls(
            host=os.environ.get("MCP_HOST", "127.0.0.1"),
            port=int(os.environ.get("MCP_PORT", "8000")),
            path=os.environ.get("MCP_PATH", "/mcp"),
        )
