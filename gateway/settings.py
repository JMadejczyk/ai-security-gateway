"""Process settings, read from ``ACL_*`` environment variables.

Secrets are required and length-checked here, so a gateway with a missing or weak signing
key refuses to start instead of issuing forgeable tokens.
"""

from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

MIN_SECRET_BYTES = 32


def _strong_secret(value: SecretStr) -> SecretStr:
    if len(value.get_secret_value().encode()) < MIN_SECRET_BYTES:
        msg = f"must be at least {MIN_SECRET_BYTES} bytes"
        raise ValueError(msg)
    return value


type Secret = Annotated[SecretStr, AfterValidator(_strong_secret)]
type Port = Annotated[int, Field(ge=1, le=65535)]


class Settings(BaseSettings):
    """Everything the process needs that is not in `policy.yaml`."""

    model_config = SettingsConfigDict(env_prefix="ACL_", frozen=True, extra="ignore")

    policy_path: Path = Path("policy.yaml")
    identities_path: Path = Path("demo/identities.yaml")
    jwt_secret: Secret  # signs and verifies agent and operator JWTs
    internal_key: Secret  # X-ACL-Principal assertions and audit payload HMACs
    demo_tokens: bool = True  # POST /auth/demo-token; ACL_DEMO_TOKENS=0 disables it
    agent_host: str = "127.0.0.1"
    agent_port: Port = 8080
    operator_host: str = "127.0.0.1"
    operator_port: Port = 9090
    audit_path: Path | None = None  # JSONL export next to the stdout audit stream
    pins_dir: Path = Path("pins")  # operator-pinned MCP tool schemas: <pins_dir>/<server>.json
    # Origins allowed to call /mcp/{server}. Agents are not browsers: by default any request
    # carrying an Origin header is refused (MCP transport security, DNS rebinding).
    mcp_allowed_origins: tuple[str, ...] = ()
    policy_watch: bool = True  # reload policy.yaml on change (POST /admin/reload always works)
    log_level: str = "info"

    @property
    def jwt_key(self) -> bytes:
        return self.jwt_secret.get_secret_value().encode()

    @property
    def internal_key_bytes(self) -> bytes:
        return self.internal_key.get_secret_value().encode()
