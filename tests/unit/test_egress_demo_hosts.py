"""``egress`` demo hosts: the overlay-only exemption for the demo's injection page.

Off by default (`Settings`, `EgressControl`). When set, a binding names one host AND one
address: only ``http://<name>:80`` passes the address check, and only while no resolved answer
differs from the bound address. The gateway shares no network with ``demo-web``, so in compose
the name does not resolve here at all; the fetch server, which connects, re-checks the same
binding and pins to it. Scheme, port and ``allow_hosts`` checks still apply.
"""

from ipaddress import ip_address

import pytest
from egress_kit import FakeResolver
from pydantic import ValidationError

from gateway.controls.egress import EgressControl
from gateway.controls.scope import CallScope, call_scope
from gateway.core.envelope import Interaction
from gateway.core.types import Action, Channel, ControlMode, Decision, SessionMode, Stage
from gateway.policy.evaluator import PrincipalContext
from gateway.policy.schema import EgressConfig
from gateway.settings import DemoHost, Settings

ALLOW = pytest.mark.control("egress", "allow")
DENY = pytest.mark.control("egress", "deny")
BLOCK = EgressConfig(mode=ControlMode.BLOCK, risk_delta=0.3)
DEMO_WEB_IP = "10.218.97.10"
DEMO = {"demo-web": ip_address(DEMO_WEB_IP)}
PAGE = "http://demo-web/q3-market-notes.html"
SECRET = "x" * 32


@pytest.fixture
def resolver() -> FakeResolver:
    # The compose case: the gateway shares no network with demo-web, so the name has no answer.
    return FakeResolver(answers={"demo-web": [], "other-host": [DEMO_WEB_IP]})


@pytest.fixture
def check(make_ctx, snapshot, resolver):
    async def run(url: str, *, demo_hosts=DEMO, cfg: EgressConfig = BLOCK):
        interaction = Interaction(
            session_id="s-test",
            principal="anna@demo",
            actor="databot",
            mode=SessionMode.INTERACTIVE,
            channel=Channel.MCP,
            server="web",
            action=Action.READ,
            resource="web:demo-web",
            payload={"name": "fetch", "arguments": {"url": url}},
            context=make_ctx(),
        )
        principal = PrincipalContext(
            principal="anna@demo", roles=("analyst",), agent="databot", mode=SessionMode.INTERACTIVE
        )
        control = EgressControl(resolver, demo_hosts=demo_hosts)
        with call_scope(CallScope(snapshot=snapshot, principal=principal)):
            return await control.evaluate(interaction, Stage.PRE, cfg)

    return run


# ------------------------------------------------------------------------------ settings


def _settings() -> Settings:
    return Settings(jwt_secret=SECRET, internal_key=SECRET)  # pyright: ignore[reportArgumentType]


def test_settings_have_no_demo_hosts_by_default(monkeypatch):
    monkeypatch.delenv("ACL_EGRESS_DEMO_HOSTS", raising=False)
    assert _settings().egress_demo_hosts == ()
    assert _settings().demo_host_bindings() == {}


def test_settings_read_name_address_bindings(monkeypatch):
    monkeypatch.setenv("ACL_EGRESS_DEMO_HOSTS", '["demo-web=10.218.97.10", "other=fd00::7"]')
    settings = _settings()
    assert settings.egress_demo_hosts == (
        DemoHost(host="demo-web", address=ip_address(DEMO_WEB_IP)),
        DemoHost(host="other", address=ip_address("fd00::7")),
    )
    assert settings.demo_host_bindings() == {
        "demo-web": ip_address(DEMO_WEB_IP),
        "other": ip_address("fd00::7"),
    }


@pytest.mark.parametrize(
    "value",
    [
        '["demo-web"]',  # a name without its address: the old, unbound format
        '["demo-web="]',
        '["=10.218.97.10"]',
        '["demo-web=not-an-ip"]',
        '["127.0.0.1=10.218.97.10"]',  # an IP literal is not a name
        '["*.internal=10.218.97.10"]',
        '["Demo-Web=10.218.97.10"]',
        '["demo-web=10.218.97.10", "demo-web=10.218.97.11"]',  # one name, two addresses
        '[""]',
    ],
)
def test_settings_refuse_unbound_or_malformed_entries(monkeypatch, value):
    monkeypatch.setenv("ACL_EGRESS_DEMO_HOSTS", value)
    with pytest.raises(ValidationError):
        _settings()


# ------------------------------------------------------------------------------ the control


@DENY
async def test_without_the_overlay_the_demo_host_is_a_private_destination(check, resolver):
    resolver.answers["demo-web"] = [DEMO_WEB_IP]
    verdict = await check(PAGE, demo_hosts={})
    assert (verdict.decision, verdict.reason_code) == (Decision.BLOCK, "egress_private_address")


@ALLOW
@pytest.mark.parametrize("answers", [[], [DEMO_WEB_IP]])
@pytest.mark.parametrize("url", [PAGE, "http://demo-web:80/"])
async def test_the_bound_origin_passes_with_its_own_reason_code(check, resolver, answers, url):
    """No answer (the compose case) or exactly the bound address: allowed and audited."""
    resolver.answers["demo-web"] = answers
    verdict = await check(url)
    assert (verdict.decision, verdict.reason_code) == (Decision.ALLOW, "egress_demo_host")
    assert verdict.risk_delta == 0.0


@DENY
@pytest.mark.parametrize(
    "answers",
    [
        ["127.0.0.1"],  # loopback
        ["169.254.169.254"],  # cloud metadata
        ["10.218.97.11"],  # another private address, same subnet
        ["172.28.0.4"],  # another internal service
        [DEMO_WEB_IP, "127.0.0.1"],  # the bound address plus another one
        ["::ffff:10.218.97.10"],  # the same IPv4 wrapped in IPv6 is not the bound address
    ],
)
async def test_the_name_resolving_anywhere_else_is_refused(check, resolver, answers):
    resolver.answers["demo-web"] = answers
    verdict = await check(PAGE)
    assert (verdict.decision, verdict.reason_code) == (Decision.BLOCK, "egress_private_address")


@DENY
@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("https://demo-web/", "egress_unresolvable"),  # https: never the demo origin
        ("http://demo-web:443/", "egress_unresolvable"),  # port 443 over http
        ("https://demo-web:443/", "egress_unresolvable"),
        ("http://other-host/", "egress_private_address"),
        ("http://demo-web.evil.example/", "egress_private_address"),
        ("http://10.218.97.10/", "egress_private_address"),  # the address as a literal
        ("http://demo-web:8080/", "egress_port_not_allowed"),
        ("ftp://demo-web/", "egress_scheme_not_allowed"),
    ],
)
async def test_the_exemption_covers_exactly_the_bound_origin(check, resolver, url, reason):
    resolver.answers["demo-web.evil.example"] = [DEMO_WEB_IP]
    verdict = await check(url)
    assert (verdict.decision, verdict.reason_code) == (Decision.BLOCK, reason)


@DENY
@pytest.mark.parametrize("answers", [[], [DEMO_WEB_IP]])
async def test_https_to_the_bound_name_never_uses_the_exemption(check, resolver, answers):
    resolver.answers["demo-web"] = answers
    verdict = await check("https://demo-web/")
    assert verdict.decision is Decision.BLOCK
    assert verdict.reason_code != "egress_demo_host"


@DENY
async def test_allow_hosts_still_applies_to_a_demo_host(check):
    cfg = EgressConfig(mode=ControlMode.BLOCK, allow_hosts=("example.com",))
    verdict = await check("http://demo-web/", cfg=cfg)
    assert (verdict.decision, verdict.reason_code) == (Decision.BLOCK, "egress_host_not_allowed")
