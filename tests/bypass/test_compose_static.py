"""Static guarantees of docker-compose.yml: who can talk to whom, and nothing that escapes Docker.

These run anywhere the docker CLI exists (no daemon, no running stack needed).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

import pytest
from compose_support import _PLACEHOLDER_SECRETS, REPO_ROOT, ComposeConfig

INTERNAL_NETWORKS = ("edge", "llm_backend", "mcp_backend", "mcp_untrusted", "state")

EXPECTED_NETWORKS = {
    "gateway": {"edge", "ops", "llm_backend", "mcp_backend", "mcp_untrusted", "state"},
    "agent": {"edge"},
    "ollama": {"llm_backend"},
    "ollama-init": {"bootstrap"},
    "models-init": {"bootstrap"},
    "mcp-postgres": {"mcp_backend"},
    "mcp-files": {"mcp_backend"},
    "postgres": {"mcp_backend"},
    "mcp-fetch": {"mcp_untrusted", "fetch_egress"},
    "redis": {"state"},
    "prometheus": {"ops"},
    "loki": {"ops"},
    "alloy": {"ops"},
    "grafana": {"ops"},
}

OBSERVABILITY = ("prometheus", "loki", "alloy", "grafana")
OBSERVABILITY_VOLUMES = {
    "prometheus": {"prometheus_data"},
    "loki": {"loki_data"},
    "alloy": {"alloy_data", "audit_log"},
    "grafana": {"grafana_data"},
}

AGENT_FORBIDDEN_ENV = re.compile(r"JWT|SECRET|INTERNAL_KEY|PASSWORD|TOKEN", re.IGNORECASE)
FROM_LINE = re.compile(
    r"^FROM\s+(?:--platform=\S+\s+)?(?P<image>\S+)(?:\s+AS\s+(?P<alias>\S+))?",
    re.IGNORECASE | re.MULTILINE,
)


def _image_tag(image: str) -> str | None:
    """Tag of an image reference, or None when it has none (digests count as pinned)."""
    if "@sha256:" in image:
        return "digest"
    name = image.rsplit("/", 1)[-1]
    return name.split(":", 1)[1] if ":" in name else None


def test_every_service_has_its_expected_networks(compose_config: ComposeConfig) -> None:
    actual = {name: compose_config.networks_of(name) for name in compose_config.services}
    assert actual == EXPECTED_NETWORKS


@pytest.mark.parametrize("network", INTERNAL_NETWORKS)
def test_backend_and_edge_networks_are_internal(
    compose_config: ComposeConfig, network: str
) -> None:
    assert compose_config.networks[network].get("internal") is True


def test_only_fetch_and_init_join_networks_with_internet(compose_config: ComposeConfig) -> None:
    egress = {
        name
        for name, spec in compose_config.networks.items()
        if not spec.get("internal") and name != "ops"
    }
    assert egress == {"fetch_egress", "bootstrap"}
    for network in egress:
        members = {s for s in compose_config.services if network in compose_config.networks_of(s)}
        expected = {"mcp-fetch"} if network == "fetch_egress" else {"ollama-init", "models-init"}
        assert members == expected


def test_only_gateway_and_grafana_publish_ports_and_only_on_loopback(
    compose_config: ComposeConfig,
) -> None:
    publishing = {name for name, svc in compose_config.services.items() if svc.get("ports")}
    assert publishing == {"gateway", "grafana"}
    grafana = cast(list[dict[str, Any]], compose_config.service("grafana")["ports"])
    assert [(p["host_ip"], p["target"]) for p in grafana] == [("127.0.0.1", 3000)]
    ports = cast(list[dict[str, Any]], compose_config.service("gateway")["ports"])
    # Host-side port numbers are overridable (ACL_*_HOST_PORT); the bind address is not.
    assert sorted((p["host_ip"], p["target"]) for p in ports) == [
        ("127.0.0.1", 8080),
        ("127.0.0.1", 9090),
    ]


def test_operator_listener_binds_the_gateway_ops_address(compose_config: ComposeConfig) -> None:
    gateway = compose_config.service("gateway")
    ops_address = gateway["networks"]["ops"]["ipv4_address"]
    environment = compose_config.environment_of("gateway")
    assert environment["ACL_OPERATOR_HOST"] == ops_address
    assert environment["ACL_OPERATOR_HOST"] != "0.0.0.0"  # noqa: S104 - asserting it is NOT bound


@pytest.mark.parametrize(
    "service",
    [
        "gateway",
        "agent",
        "ollama",
        "mcp-postgres",
        "mcp-files",
        "mcp-fetch",
        "postgres",
        "redis",
        "models-init",
        *OBSERVABILITY,
    ],
)
def test_no_service_escapes_docker_isolation(compose_config: ComposeConfig, service: str) -> None:
    spec = compose_config.service(service)
    assert not spec.get("privileged", False)
    assert not spec.get("cap_add")
    assert not spec.get("devices")
    for namespace in ("network_mode", "pid", "ipc", "userns_mode", "uts"):
        assert spec.get(namespace) != "host", namespace
    for volume in cast(list[dict[str, Any]], spec.get("volumes", [])):
        assert "docker.sock" not in str(volume.get("source", ""))
        assert str(volume.get("source", "")) not in {"/", "/var/run", "/run"}


def test_agent_holds_no_secrets_or_mounts(compose_config: ComposeConfig) -> None:
    agent = compose_config.service("agent")
    leaked = [
        key for key in compose_config.environment_of("agent") if AGENT_FORBIDDEN_ENV.search(key)
    ]
    assert leaked == []
    assert not agent.get("env_file")
    assert not agent.get("secrets")
    assert not agent.get("volumes")


def test_signing_key_reaches_only_the_gateway(compose_config: ComposeConfig) -> None:
    holders = {
        s for s in compose_config.services if "ACL_JWT_SECRET" in compose_config.environment_of(s)
    }
    assert holders == {"gateway"}


def test_internal_key_reaches_only_gateway_and_mcp_postgres(compose_config: ComposeConfig) -> None:
    holders = {
        s for s in compose_config.services if "ACL_INTERNAL_KEY" in compose_config.environment_of(s)
    }
    assert holders == {"gateway", "mcp-postgres"}


def test_pulled_images_are_pinned(compose_config: ComposeConfig) -> None:
    for name, spec in compose_config.services.items():
        if "build" in spec:
            continue
        tag = _image_tag(str(spec["image"]))
        assert tag not in {None, "latest"}, f"{name} uses unpinned image {spec['image']}"


def test_built_images_use_pinned_base_images(compose_config: ComposeConfig) -> None:
    for name, spec in compose_config.services.items():
        build = cast(dict[str, Any] | None, spec.get("build"))
        if build is None:
            continue
        dockerfile = Path(build["context"]) / build.get("dockerfile", "Dockerfile")
        assert dockerfile.resolve().is_relative_to(REPO_ROOT), dockerfile
        stages: set[str] = set()
        for match in FROM_LINE.finditer(dockerfile.read_text()):
            image, alias = match.group("image"), match.group("alias")
            if image not in stages:
                tag = _image_tag(image)
                assert tag not in {None, "latest"}, f"{name}: {image} in {dockerfile}"
            if alias:
                stages.add(alias)


def test_state_network_joins_only_gateway_and_redis(compose_config: ComposeConfig) -> None:
    members = {s for s in compose_config.services if "state" in compose_config.networks_of(s)}
    assert members == {"gateway", "redis"}


def test_redis_password_reaches_only_redis_and_gateway(compose_config: ComposeConfig) -> None:
    holders = {
        s
        for s in compose_config.services
        if "ACL_REDIS_PASSWORD" in compose_config.environment_of(s)
    }
    assert holders == {"gateway", "redis"}


def test_redis_is_hardened_and_on_the_bsd_licensed_line(compose_config: ComposeConfig) -> None:
    redis = compose_config.service("redis")
    assert str(redis["image"]).startswith("redis:7.2.")  # 7.4+ is RSALv2/SSPL, 8.x adds AGPL
    assert redis.get("read_only") is True
    assert redis.get("cap_drop") == ["ALL"]
    assert "no-new-privileges:true" in redis.get("security_opt", [])
    assert redis.get("user") == "redis"
    assert not redis.get("ports")
    command = " ".join(cast(list[str], redis["command"]))
    assert "requirepass" in command
    assert "placeholder-redis" not in command  # the secret is read at runtime, not interpolated


def test_gateway_uses_redis_for_budgets(compose_config: ComposeConfig) -> None:
    environment = compose_config.environment_of("gateway")
    assert environment["ACL_BUDGET_STORE"] == "redis"
    assert environment["ACL_REDIS_URL"] == "redis://redis:6379/0"


def _volume_mounts(compose_config: ComposeConfig, service: str) -> dict[str, dict[str, Any]]:
    mounts = cast(list[dict[str, Any]], compose_config.service(service).get("volumes", []))
    return {str(m["source"]): m for m in mounts if m.get("type") == "volume"}


def test_injection_model_is_fetched_on_bootstrap_and_mounted_read_only(
    compose_config: ComposeConfig,
) -> None:
    """models-init (internet, bootstrap only) writes the classifier; the gateway only reads it."""
    holders = {
        s for s in compose_config.services if "injection_model" in _volume_mounts(compose_config, s)
    }
    assert holders == {"gateway", "models-init"}
    gateway_mount = _volume_mounts(compose_config, "gateway")["injection_model"]
    assert gateway_mount.get("read_only") is True
    assert compose_config.environment_of("gateway")["ACL_MODELS_DIR"] == gateway_mount["target"]
    init = compose_config.service("models-init")
    assert compose_config.networks_of("models-init") == {"bootstrap"}
    assert {"init", "models-init"} <= set(init.get("profiles", []))
    assert init.get("read_only") is True
    assert init.get("cap_drop") == ["ALL"]
    assert not init.get("ports")
    assert not [
        k for k in compose_config.environment_of("models-init") if AGENT_FORBIDDEN_ENV.search(k)
    ]
    assert "gateway.injection.fetch" in " ".join(cast(list[str], init["command"]))


# --------------------------------------------------------------------------- observability


def test_ops_members_are_the_gateway_and_the_observability_services(
    compose_config: ComposeConfig,
) -> None:
    members = {s for s in compose_config.services if "ops" in compose_config.networks_of(s)}
    assert members == {"gateway", *OBSERVABILITY}


@pytest.mark.parametrize("service", OBSERVABILITY)
def test_observability_services_are_hardened(compose_config: ComposeConfig, service: str) -> None:
    spec = compose_config.service(service)
    assert spec.get("cap_drop") == ["ALL"]
    assert "no-new-privileges:true" in spec.get("security_opt", [])
    assert spec.get("read_only") is True
    assert str(spec.get("user", "")) not in {"0", "root", "0:0"}
    assert compose_config.networks_of(service) == {"ops"}
    tag = _image_tag(str(spec["image"]))
    assert tag is not None
    assert re.fullmatch(r"v?\d+\.\d+\.\d+", tag), f"{service}: {spec['image']} not an exact tag"


@pytest.mark.parametrize("service", OBSERVABILITY)
def test_observability_services_mount_only_their_volumes_and_config(
    compose_config: ComposeConfig, service: str
) -> None:
    mounts = cast(list[dict[str, Any]], compose_config.service(service).get("volumes", []))
    volumes = {str(m["source"]) for m in mounts if m.get("type") == "volume"}
    assert volumes == OBSERVABILITY_VOLUMES[service]
    for mount in mounts:
        if mount.get("type") == "bind":  # configuration from the repo, never writable
            assert mount.get("read_only") is True, mount
            assert Path(str(mount["source"])).resolve().is_relative_to(REPO_ROOT), mount
        assert "docker.sock" not in str(mount.get("source", "")) + str(mount.get("target", ""))


def test_alloy_has_no_docker_socket_and_reads_the_audit_log_only(
    compose_config: ComposeConfig,
) -> None:
    alloy = compose_config.service("alloy")
    rendered = json.dumps(alloy)
    assert "docker.sock" not in rendered
    assert "/var/run" not in rendered
    audit = _volume_mounts(compose_config, "alloy")["audit_log"]
    assert audit.get("read_only") is True
    assert audit["target"] == "/var/log/acl"


def test_audit_log_volume_is_operator_only(compose_config: ComposeConfig) -> None:
    """The gateway writes the JSONL export; alloy reads it; nothing else mounts it."""
    holders = {
        s for s in compose_config.services if "audit_log" in _volume_mounts(compose_config, s)
    }
    assert holders == {"gateway", "alloy"}
    gateway_mount = _volume_mounts(compose_config, "gateway")["audit_log"]
    assert not gateway_mount.get("read_only")
    audit_path = compose_config.environment_of("gateway")["ACL_AUDIT_PATH"]
    assert audit_path is not None
    assert Path(audit_path).parent == Path(gateway_mount["target"])


def test_grafana_secret_reaches_only_grafana(compose_config: ComposeConfig) -> None:
    placeholder = _PLACEHOLDER_SECRETS["ACL_GRAFANA_ADMIN_PASSWORD"]
    holders = {s for s, spec in compose_config.services.items() if placeholder in json.dumps(spec)}
    assert holders == {"grafana"}
    environment = compose_config.environment_of("grafana")
    assert environment["GF_SECURITY_ADMIN_PASSWORD"] == placeholder
    assert environment["GF_AUTH_ANONYMOUS_ENABLED"] == "false"
    assert environment["GF_USERS_ALLOW_SIGN_UP"] == "false"


@pytest.mark.parametrize("service", ["prometheus", "loki", "alloy"])
def test_observability_services_hold_no_secrets(
    compose_config: ComposeConfig, service: str
) -> None:
    rendered = json.dumps(compose_config.service(service))
    assert "placeholder" not in rendered
    assert not compose_config.service(service).get("env_file")


def test_observability_image_versions_are_recorded_in_licenses(
    compose_config: ComposeConfig,
) -> None:
    """LICENSES.md records these exact versions; a bump must update it."""
    licenses = (REPO_ROOT / "LICENSES.md").read_text()
    for service in OBSERVABILITY:
        image = str(compose_config.service(service)["image"])
        assert image in licenses, f"{image} is not recorded in LICENSES.md"


def test_ollama_keeps_a_cache_slot_for_the_agent_and_one_for_the_judge(
    compose_config: ComposeConfig,
) -> None:
    """Agent and judge share one model: with one slot each call evicts the other's cached
    prompt and every judge call re-reads its whole prompt on CPU (12-20 s instead of ~6 s)."""
    environment = compose_config.environment_of("ollama")
    assert environment["OLLAMA_NUM_PARALLEL"] == "2"
    assert environment["OLLAMA_MAX_LOADED_MODELS"] == "1"  # bounded memory: one model, 2 slots
    assert int(environment["OLLAMA_CONTEXT_LENGTH"] or "0") in range(1024, 8193)
