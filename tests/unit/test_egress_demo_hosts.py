"""``egress`` demo hosts: the overlay-only exemption for the demo's injection page.

Off by default (`Settings`, `EgressControl`); when set, exactly the listed names skip the
address check, with their own reason code, and the scheme, port and ``allow_hosts`` checks
still apply to them.
"""

import pytest
from egress_kit import FakeResolver
from pydantic import ValidationError

from gateway.controls.egress import EgressControl
from gateway.controls.scope import CallScope, call_scope
from gateway.core.envelope import Interaction
from gateway.core.types import Action, Channel, ControlMode, Decision, SessionMode, Stage
from gateway.policy.evaluator import PrincipalContext
from gateway.policy.schema import EgressConfig
from gateway.settings import Settings

ALLOW = pytest.mark.control("egress", "allow")
DENY = pytest.mark.control("egress", "deny")
BLOCK = EgressConfig(mode=ControlMode.BLOCK, risk_delta=0.3)
DEMO_WEB_IP = "172.30.99.2"
SECRET = "x" * 32


@pytest.fixture
def resolver() -> FakeResolver:
    return FakeResolver(answers={"demo-web": [DEMO_WEB_IP], "other-host": [DEMO_WEB_IP]})


@pytest.fixture
def check(make_ctx, snapshot, resolver):
    async def run(url: str, *, demo_hosts: frozenset[str], cfg: EgressConfig = BLOCK):
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


def test_settings_have_no_demo_hosts_by_default(monkeypatch):
    monkeypatch.delenv("ACL_EGRESS_DEMO_HOSTS", raising=False)
    assert Settings(jwt_secret=SECRET, internal_key=SECRET).egress_demo_hosts == ()  # pyright: ignore[reportArgumentType]


def test_settings_read_demo_hosts_as_a_json_list(monkeypatch):
    monkeypatch.setenv("ACL_EGRESS_DEMO_HOSTS", '["demo-web"]')
    settings = Settings(jwt_secret=SECRET, internal_key=SECRET)  # pyright: ignore[reportArgumentType]
    assert settings.egress_demo_hosts == ("demo-web",)


@pytest.mark.parametrize("value", ['["127.0.0.1"]', '["*.internal"]', '["Demo-Web"]', '[""]'])
def test_settings_refuse_ip_literals_globs_and_odd_names(monkeypatch, value):
    monkeypatch.setenv("ACL_EGRESS_DEMO_HOSTS", value)
    with pytest.raises(ValidationError):
        Settings(jwt_secret=SECRET, internal_key=SECRET)  # pyright: ignore[reportArgumentType]


@DENY
async def test_without_the_overlay_the_demo_host_is_a_private_destination(check):
    verdict = await check("http://demo-web/q3-market-notes.html", demo_hosts=frozenset())
    assert (verdict.decision, verdict.reason_code) == (Decision.BLOCK, "egress_private_address")


@ALLOW
async def test_a_listed_demo_host_passes_with_its_own_reason_code(check, resolver):
    verdict = await check(
        "http://demo-web/q3-market-notes.html", demo_hosts=frozenset({"demo-web"})
    )
    assert (verdict.decision, verdict.reason_code) == (Decision.ALLOW, "egress_demo_host")
    assert resolver.calls == []  # the gateway shares no network with it: nothing to resolve


@DENY
@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("http://other-host/", "egress_private_address"),
        ("http://demo-web.evil.example/", "egress_private_address"),
        ("http://demo-web:8080/", "egress_port_not_allowed"),
        ("ftp://demo-web/", "egress_scheme_not_allowed"),
    ],
)
async def test_only_the_exact_name_on_an_allowed_port_is_exempt(check, resolver, url, reason):
    resolver.answers["demo-web.evil.example"] = [DEMO_WEB_IP]
    verdict = await check(url, demo_hosts=frozenset({"demo-web"}))
    assert (verdict.decision, verdict.reason_code) == (Decision.BLOCK, reason)


@DENY
async def test_allow_hosts_still_applies_to_a_demo_host(check):
    cfg = EgressConfig(mode=ControlMode.BLOCK, allow_hosts=("example.com",))
    verdict = await check("http://demo-web/", demo_hosts=frozenset({"demo-web"}), cfg=cfg)
    assert (verdict.decision, verdict.reason_code) == (Decision.BLOCK, "egress_host_not_allowed")
