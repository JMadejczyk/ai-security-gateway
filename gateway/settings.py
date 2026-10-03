"""Process settings, read from ``ACL_*`` environment variables.

Secrets are required and length-checked here, so a gateway with a missing or weak signing
key refuses to start instead of issuing forgeable tokens.
"""

from pathlib import Path
from typing import Annotated, Literal

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
# A single lower-case DNS name with a letter in its first label: never an IP literal or a glob.
type DemoHost = Annotated[
    str, Field(pattern=r"^[a-z][a-z0-9-]*(?:\.[a-z0-9][a-z0-9-]*)*$", max_length=253)
]


class Settings(BaseSettings):
    """Everything the process needs that is not in `policy.yaml`."""

    model_config = SettingsConfigDict(env_prefix="ACL_", frozen=True, extra="ignore")

    policy_path: Path = Path("config/policy.yaml")
    identities_path: Path = Path("demo/identities.yaml")
    jwt_secret: Secret  # signs and verifies agent and operator JWTs
    internal_key: Secret  # X-ACL-Principal assertions and audit payload HMACs
    demo_tokens: bool = True  # POST /auth/demo-token; ACL_DEMO_TOKENS=0 disables it
    agent_host: str = "127.0.0.1"
    agent_port: Port = 8080
    operator_host: str = "127.0.0.1"
    operator_port: Port = 9090
    audit_path: Path | None = None  # JSONL export next to the stdout audit stream
    # The export is never-renamed segments next to audit_path (audit-<time>-<n>.jsonl): a new
    # one every audit_max_bytes, audit_backups old ones kept (`telemetry.SegmentedFileHandler`).
    audit_max_bytes: int = Field(default=50 * 1024 * 1024, ge=64 * 1024)
    audit_backups: int = Field(default=4, ge=1, le=100)
    pins_dir: Path = Path("pins")  # operator-pinned MCP tool schemas: <pins_dir>/<server>.json
    # Origins allowed to call /mcp/{server}. Agents are not browsers: by default any request
    # carrying an Origin header is refused (MCP transport security, DNS rebinding).
    mcp_allowed_origins: tuple[str, ...] = ()
    policy_watch: bool = True  # reload policy.yaml on change (POST /admin/reload always works)
    # Budget counters. `redis` is the only production store: when it is unreachable every
    # budget-limited call fails closed (503). `memory` (one process, lost on restart) is for
    # tests and local development and is only ever used when chosen explicitly.
    budget_store: Literal["redis", "memory"] = "redis"
    redis_url: str = "redis://127.0.0.1:6379/0"
    redis_password: SecretStr | None = None  # `requirepass` of the Redis on the `state` network
    # Session state, loop counters, approvals and the kill switch (`gateway.state_redis`).
    # None = the same store as `budget_store`. `redis` fails closed (503) while Redis is down.
    session_store: Literal["redis", "memory"] | None = None
    # How long a call waits for another call of its session to finish before it is refused
    # with 429 `session_busy` (calls in one session are serialized across gateways).
    session_lock_wait_s: float = Field(default=30.0, gt=0.0, le=600.0)
    # Prompt-injection classifier (`gateway.injection`). `models_dir` holds the files the pinned
    # manifest names; the gateway verifies every SHA-256 and refuses to start on any mismatch.
    # `disabled` (development only) loads no model: prompt_injection and tool_poisoning then
    # fail closed on every text they would classify (`classifier_unavailable`).
    models_dir: Path = Path("models/cache")
    injection_classifier: Literal["onnx", "disabled"] = "onnx"
    classifier_threads: int = Field(default=4, ge=1, le=64)  # intra-op threads per inference
    classifier_workers: int = Field(default=2, ge=1, le=64)  # inferences running at once
    # Exact host names `egress` lets through without the public-address check (reason code
    # `egress_demo_host`). Empty by default; only the demo overlay (demo/compose.demo.yml) sets
    # it, as a JSON list: ACL_EGRESS_DEMO_HOSTS='["demo-web"]'.
    egress_demo_hosts: tuple[DemoHost, ...] = ()
    log_level: str = "info"

    @property
    def jwt_key(self) -> bytes:
        return self.jwt_secret.get_secret_value().encode()

    @property
    def internal_key_bytes(self) -> bytes:
        return self.internal_key.get_secret_value().encode()
