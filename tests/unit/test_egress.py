"""``egress`` on its own: destinations of an ``adapter: http`` server, with an injected resolver."""

from ipaddress import ip_address
from typing import Any

import pytest
from egress_kit import PUBLIC_IP, FakeResolver

from gateway.controls.egress import EgressControl, is_public_address
from gateway.controls.scope import CallScope, call_scope
from gateway.core.envelope import Interaction
from gateway.core.types import Action, Channel, ControlMode, Decision, SessionMode, Stage
from gateway.policy.evaluator import PrincipalContext
from gateway.policy.schema import EgressConfig

ALLOW = pytest.mark.control("egress", "allow")
DENY = pytest.mark.control("egress", "deny")
BLOCK = EgressConfig(mode=ControlMode.BLOCK, risk_delta=0.3)


@pytest.fixture
def resolver() -> FakeResolver:
    return FakeResolver()


@pytest.fixture
def check(make_ctx, snapshot, resolver):
    """Evaluate one ``fetch`` (or another server's tool) under a call scope of the root policy."""

    async def run(
        url: str | None,
        cfg: EgressConfig = BLOCK,
        *,
        server: str = "web",
        scoped: bool = True,
    ):
        arguments: dict[str, Any] = {} if url is None else {"url": url}
        interaction = Interaction(
            session_id="s-test",
            principal="anna@demo",
            actor="databot",
            mode=SessionMode.INTERACTIVE,
            channel=Channel.MCP,
            server=server,
            action=Action.READ,
            resource="web:example.com",
            payload={"name": "fetch", "arguments": arguments},
            context=make_ctx(),
        )
        control = EgressControl(resolver)
        if not scoped:
            return await control.evaluate(interaction, Stage.PRE, cfg)
        principal = PrincipalContext(
            principal="anna@demo", roles=("analyst",), agent="databot", mode=SessionMode.INTERACTIVE
        )
        with call_scope(CallScope(snapshot=snapshot, principal=principal)):
            return await control.evaluate(interaction, Stage.PRE, cfg)

    return run


@ALLOW
@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/",
        "http://example.com/page?q=1",
        "https://example.com:443/",
        "https://93.184.215.14/",
        "https://[2606:4700::1111]/",
    ],
)
async def test_public_destinations_pass(check, url):
    verdict = await check(url)
    assert (verdict.decision, verdict.reason_code) == (Decision.ALLOW, "egress_allowed")
    assert verdict.risk_delta == 0.0


@DENY
@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata (link-local)
        "http://127.0.0.1/",
        "http://10.0.0.5/",
        "http://172.16.3.4/",
        "http://192.168.1.1/",
        "http://100.64.0.1/",  # CGNAT
        "http://0.0.0.0/",
        "http://224.0.0.1/",  # multicast
        "http://240.0.0.1/",  # reserved
        "http://[::1]/",
        "http://[fd00::1]/",  # ULA
        "http://[fe80::1]/",  # link-local
        "http://[::ffff:169.254.169.254]/",  # IPv4-mapped metadata
        "http://[2002:a9fe:a9fe::1]/",  # 6to4 wrapping 169.254.169.254
        "http://[2001:0:4136:e378:8000:63bf:f5fe:fffe]/",  # Teredo wrapping 10.1.0.1
    ],
)
async def test_non_public_ip_literals_are_refused(check, resolver, url):
    verdict = await check(url)
    assert (verdict.decision, verdict.reason_code) == (Decision.BLOCK, "egress_private_address")
    assert verdict.risk_delta == pytest.approx(0.3)
    assert resolver.calls == []  # a literal is never resolved


@DENY
@pytest.mark.parametrize(
    "answers",
    [["127.0.0.1"], ["169.254.169.254"], ["10.1.2.3"], ["::1"], [PUBLIC_IP, "192.168.0.10"]],
    ids=["loopback", "metadata", "rfc1918", "ipv6-loopback", "one-of-two-private"],
)
async def test_a_host_resolving_to_any_private_address_is_refused(check, resolver, answers):
    resolver.answers["intranet.example"] = answers
    verdict = await check("https://intranet.example/admin")
    assert (verdict.decision, verdict.reason_code) == (Decision.BLOCK, "egress_private_address")
    assert resolver.calls == [("intranet.example", 443)]


@ALLOW
async def test_a_host_resolving_only_to_public_addresses_passes(check, resolver):
    resolver.answers["cdn.example"] = [PUBLIC_IP, "2606:4700::1111"]
    verdict = await check("http://cdn.example:80/x")
    assert verdict.decision is Decision.ALLOW
    assert resolver.calls == [("cdn.example", 80)]


@DENY
async def test_an_unresolvable_host_fails_closed(check, resolver):
    resolver.answers["nowhere.example"] = []
    verdict = await check("https://nowhere.example/")
    assert (verdict.decision, verdict.reason_code) == (Decision.BLOCK, "egress_unresolvable")


@DENY
async def test_a_stalled_resolver_times_out_closed(check, resolver):
    resolver.stall = True
    cfg = EgressConfig(mode=ControlMode.BLOCK, resolve_timeout_s=0.01)
    verdict = await check("https://slow.example/", cfg)
    assert (verdict.decision, verdict.reason_code) == (Decision.BLOCK, "egress_unresolvable")


@pytest.mark.parametrize(
    ("url", "cfg", "decision", "reason"),
    [
        pytest.param(
            "https://example.com:8443/", BLOCK, Decision.BLOCK, "egress_port_not_allowed",
            marks=DENY, id="port-8443-default-ports",
        ),
        pytest.param(
            "http://example.com:22/", BLOCK, Decision.BLOCK, "egress_port_not_allowed",
            marks=DENY, id="port-22",
        ),
        pytest.param(
            "https://example.com:8443/",
            EgressConfig(mode=ControlMode.BLOCK, allowed_ports=(443, 8443)),
            Decision.ALLOW, "egress_allowed", marks=ALLOW, id="port-8443-configured",
        ),
    ],
)  # fmt: skip
async def test_ports_follow_allowed_ports(check, url, cfg, decision, reason):
    verdict = await check(url, cfg)
    assert (verdict.decision, verdict.reason_code) == (decision, reason)


ALLOWLIST = EgressConfig(mode=ControlMode.BLOCK, allow_hosts=("api.stripe.com", "*.example.com"))


@pytest.mark.parametrize(
    ("url", "decision", "reason"),
    [
        pytest.param("https://api.stripe.com/v1", Decision.ALLOW, "egress_allowed", marks=ALLOW),
        pytest.param("https://docs.example.com/", Decision.ALLOW, "egress_allowed", marks=ALLOW),
        pytest.param("https://example.com/", Decision.BLOCK, "egress_host_not_allowed", marks=DENY),
        pytest.param(
            "https://pastebin.com/raw/x", Decision.BLOCK, "egress_host_not_allowed", marks=DENY
        ),
    ],
)
async def test_allow_hosts_restricts_destinations(check, url, decision, reason):
    verdict = await check(url, ALLOWLIST)
    assert (verdict.decision, verdict.reason_code) == (decision, reason)


async def test_an_allowlisted_host_must_still_be_public(check, resolver):
    resolver.answers["internal.example.com"] = ["10.0.0.7"]
    verdict = await check("https://internal.example.com/", ALLOWLIST)
    assert verdict.reason_code == "egress_private_address"


HOLD = EgressConfig(mode=ControlMode.REQUIRE_APPROVAL, allow_hosts=("*.com", "169.254.169.254"))


@pytest.mark.control("egress", "require_approval")
@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("http://169.254.169.254/", "egress_private_address"),
        ("https://example.com:8443/", "egress_port_not_allowed"),
        ("https://example.org/", "egress_host_not_allowed"),
    ],
)
async def test_require_approval_mode_holds_what_an_operator_can_judge(check, url, reason):
    verdict = await check(url, HOLD)
    assert (verdict.decision, verdict.reason_code) == (Decision.REQUIRE_APPROVAL, reason)


@DENY
async def test_require_approval_mode_still_blocks_what_cannot_be_checked(check, resolver):
    resolver.answers["nowhere.example"] = []
    cfg = EgressConfig(mode=ControlMode.REQUIRE_APPROVAL)
    assert (await check("https://nowhere.example/", cfg)).decision is Decision.BLOCK
    assert (await check(None, cfg)).decision is Decision.BLOCK  # no url argument


@ALLOW
@pytest.mark.parametrize("server", ["sales_db", "reports"])
async def test_servers_without_the_http_adapter_are_not_its_concern(check, resolver, server):
    verdict = await check("http://127.0.0.1/", server=server)
    assert (verdict.decision, verdict.reason_code) == (Decision.ALLOW, "egress_not_applicable")
    assert resolver.calls == []


@DENY
async def test_no_call_scope_fails_closed(check):
    verdict = await check("https://example.com/", scoped=False)
    assert (verdict.decision, verdict.reason_code) == (Decision.BLOCK, "egress_unverifiable")


@DENY
@pytest.mark.parametrize("url", ["ftp://example.com/x", "file:///etc/passwd"])
async def test_only_http_schemes_pass(check, url):
    verdict = await check(url)
    assert verdict.decision is Decision.BLOCK
    assert verdict.reason_code in {"egress_scheme_not_allowed", "egress_unverifiable"}


@pytest.mark.parametrize(
    ("address", "public"),
    [
        ("8.8.8.8", True),
        ("2606:4700::1111", True),
        ("169.254.169.254", False),
        ("::ffff:10.0.0.1", False),
        ("192.0.2.1", False),  # TEST-NET documentation range
    ],
)
def test_is_public_address(address, public):
    assert is_public_address(ip_address(address)) is public
