from __future__ import annotations

import asyncio
from collections.abc import Sequence
from ipaddress import IPv4Address, IPv6Address, ip_address

import httpx
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from acl_demo_mcp.fetch_server import (
    DisallowedDestinationError,
    Fetcher,
    FetchSettings,
    InvalidDemoHostsError,
    UnresolvableHostError,
    UnsupportedUrlError,
    is_public_address,
)

IPAddress = IPv4Address | IPv6Address
PUBLIC_V4 = "93.184.215.14"
PUBLIC_V6 = "2606:4700:4700::1111"


class FakeResolver:
    """Answers from a script: one answer list per call (the last one repeats)."""

    def __init__(self, *answers: Sequence[str]) -> None:
        self._answers = [[ip_address(a) for a in answer] for answer in answers]
        self.calls: list[tuple[str, int]] = []

    async def __call__(self, host: str, port: int) -> Sequence[IPAddress]:
        self.calls.append((host, port))
        return self._answers[min(len(self.calls), len(self._answers)) - 1]


class RecordingTransport(httpx.MockTransport):
    def __init__(self, body: bytes = b"hello") -> None:
        self.requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(200, content=body, headers={"content-type": "text/plain"})

        super().__init__(handler)


def _fetch(fetcher: Fetcher, url: str) -> str:
    return asyncio.run(fetcher.fetch(url))


@pytest.fixture
def transport() -> RecordingTransport:
    return RecordingTransport()


@pytest.mark.parametrize("address", [PUBLIC_V4, "1.1.1.1", PUBLIC_V6])
def test_public_addresses_are_allowed(address: str) -> None:
    assert is_public_address(ip_address(address))


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",  # loopback
        "127.8.9.10",
        "10.0.0.1",  # RFC 1918
        "172.16.0.1",
        "172.29.90.10",  # the gateway's operator address on `ops`
        "192.168.1.1",
        "100.64.0.1",  # CGNAT
        "169.254.169.254",  # cloud metadata (link-local)
        "0.0.0.0",  # noqa: S104 - unspecified, must be rejected
        "224.0.0.1",  # multicast
        "239.255.255.250",
        "255.255.255.255",  # broadcast
        "198.18.0.1",  # benchmarking
        "::1",
        "::",
        "fe80::1",  # link-local
        "fc00::1",  # ULA
        "fd12:3456::1",
        "ff02::1",  # multicast
        "::ffff:127.0.0.1",  # IPv4-mapped variants
        "::ffff:169.254.169.254",
        "::ffff:10.0.0.1",
        "::ffff:172.29.90.10",
        "2002:7f00:1::1",  # 6to4 wrapping 127.0.0.1
        "2002:a9fe:a9fe::1",  # 6to4 wrapping 169.254.169.254
    ],
)
def test_internal_addresses_are_rejected(address: str) -> None:
    assert not is_public_address(ip_address(address))


def test_connects_to_the_validated_ip_with_original_host_and_sni(
    transport: RecordingTransport,
) -> None:
    resolver = FakeResolver([PUBLIC_V4])
    fetcher = Fetcher(FetchSettings(), resolver=resolver, transport=transport)

    assert _fetch(fetcher, "https://example.test/page?q=1") == "hello"

    (request,) = transport.requests
    assert request.url.host == PUBLIC_V4
    assert request.url.port is None
    assert request.url.path == "/page"
    assert request.url.query == b"q=1"
    assert request.headers["host"] == "example.test"
    assert request.extensions["sni_hostname"] == "example.test"
    assert resolver.calls == [("example.test", 443)]


def test_ipv6_answer_is_pinned_too(transport: RecordingTransport) -> None:
    fetcher = Fetcher(FetchSettings(), resolver=FakeResolver([PUBLIC_V6]), transport=transport)
    _fetch(fetcher, "http://example.test/")
    (request,) = transport.requests
    assert request.url.host == PUBLIC_V6
    assert request.headers["host"] == "example.test"


@pytest.mark.parametrize(
    "answer",
    [["10.1.2.3"], ["127.0.0.1"], ["169.254.169.254"], ["::ffff:127.0.0.1"], ["fd00::5"]],
)
def test_name_resolving_to_internal_address_is_rejected(
    transport: RecordingTransport, answer: list[str]
) -> None:
    fetcher = Fetcher(FetchSettings(), resolver=FakeResolver(answer), transport=transport)
    with pytest.raises(DisallowedDestinationError):
        _fetch(fetcher, "http://innocent.example.test/")
    assert transport.requests == []


def test_any_internal_address_among_the_answers_rejects(transport: RecordingTransport) -> None:
    fetcher = Fetcher(
        FetchSettings(), resolver=FakeResolver([PUBLIC_V4, "10.0.0.7"]), transport=transport
    )
    with pytest.raises(DisallowedDestinationError):
        _fetch(fetcher, "http://mixed.example.test/")
    assert transport.requests == []


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",
        "http://127.0.0.1/",
        "http://[::1]/",
        "http://[::ffff:169.254.169.254]/",
        "http://172.29.90.10/healthz",
    ],
)
def test_internal_ip_literals_are_rejected_without_resolving(
    transport: RecordingTransport, url: str
) -> None:
    resolver = FakeResolver([PUBLIC_V4])
    fetcher = Fetcher(FetchSettings(), resolver=resolver, transport=transport)
    with pytest.raises(DisallowedDestinationError):
        _fetch(fetcher, url)
    assert resolver.calls == []
    assert transport.requests == []


def test_dns_rebinding_cannot_redirect_the_connection(transport: RecordingTransport) -> None:
    # First answer is public (passes validation), every later answer is internal.
    resolver = FakeResolver([PUBLIC_V4], ["127.0.0.1"])
    fetcher = Fetcher(FetchSettings(), resolver=resolver, transport=transport)

    _fetch(fetcher, "http://rebind.example.test/")

    assert len(resolver.calls) == 1
    (request,) = transport.requests
    assert request.url.host == PUBLIC_V4


@pytest.mark.parametrize(
    "url",
    ["http://example.test:8080/", "https://example.test:8443/", "http://example.test:5432/"],
)
def test_non_web_ports_are_rejected(transport: RecordingTransport, url: str) -> None:
    fetcher = Fetcher(FetchSettings(), resolver=FakeResolver([PUBLIC_V4]), transport=transport)
    with pytest.raises(UnsupportedUrlError):
        _fetch(fetcher, url)
    assert transport.requests == []


@pytest.mark.parametrize("url", ["http://example.test:80/", "https://example.test:443/"])
def test_default_web_ports_are_allowed(transport: RecordingTransport, url: str) -> None:
    fetcher = Fetcher(FetchSettings(), resolver=FakeResolver([PUBLIC_V4]), transport=transport)
    assert _fetch(fetcher, url) == "hello"


def test_unresolvable_host_is_a_tool_error(transport: RecordingTransport) -> None:
    fetcher = Fetcher(FetchSettings(), resolver=FakeResolver([]), transport=transport)
    with pytest.raises(UnresolvableHostError):
        _fetch(fetcher, "http://nowhere.example.test/")


def test_rejections_are_tool_errors() -> None:
    for error in (DisallowedDestinationError, UnresolvableHostError, UnsupportedUrlError):
        assert issubclass(error, ToolError)


# ------------------------------------------------------------------ demo hosts (overlay only)

DEMO_WEB_IP = "10.218.97.10"
DEMO = {"demo-web": ip_address(DEMO_WEB_IP)}
PAGE = "http://demo-web/q3-market-notes.html"


def test_demo_hosts_are_off_by_default() -> None:
    assert FetchSettings().demo_hosts == {}
    assert FetchSettings.from_env({}).demo_hosts == {}
    assert FetchSettings.from_env({"ACL_FETCH_DEMO_HOSTS": ""}).demo_hosts == {}


def test_demo_hosts_env_binds_each_name_to_one_address() -> None:
    settings = FetchSettings.from_env(
        {"ACL_FETCH_DEMO_HOSTS": " demo-web=10.218.97.10 , other=fd00::7 "}
    )
    assert settings.demo_hosts == {
        "demo-web": ip_address("10.218.97.10"),
        "other": ip_address("fd00::7"),
    }


@pytest.mark.parametrize(
    "raw",
    [
        "demo-web",  # a name without its address: the old, unbound format
        "demo-web=",
        "=10.218.97.10",
        "demo-web=not-an-ip",
        "Demo-Web=10.218.97.10",  # names are lower case
        "10.0.0.1=10.218.97.10",  # an IP literal is not a name
        "*.web=10.218.97.10",
        "demo-web=10.218.97.10,demo-web=10.218.97.11",  # one name, two addresses
        "demo-web=10.218.97.10,,",
    ],
)
def test_malformed_demo_hosts_refuse_to_start(raw: str) -> None:
    with pytest.raises(InvalidDemoHostsError):
        FetchSettings.from_env({"ACL_FETCH_DEMO_HOSTS": raw})


def test_without_the_overlay_the_demo_host_is_refused(transport: RecordingTransport) -> None:
    fetcher = Fetcher(FetchSettings(), resolver=FakeResolver([DEMO_WEB_IP]), transport=transport)
    with pytest.raises(DisallowedDestinationError):
        _fetch(fetcher, PAGE)
    assert transport.requests == []


def test_the_bound_address_on_http_80_is_fetched_pinned_to_it(
    transport: RecordingTransport,
) -> None:
    settings = FetchSettings(demo_hosts=DEMO)
    fetcher = Fetcher(settings, resolver=FakeResolver([DEMO_WEB_IP]), transport=transport)
    assert _fetch(fetcher, PAGE) == "hello"
    assert _fetch(fetcher, "http://demo-web:80/") == "hello"
    first, second = transport.requests
    assert (first.url.host, first.url.scheme, first.url.port) == (DEMO_WEB_IP, "http", None)
    assert first.headers["host"] == "demo-web"
    assert second.url.host == DEMO_WEB_IP


@pytest.mark.parametrize(
    "answer",
    [
        ["127.0.0.1"],  # loopback
        ["169.254.169.254"],  # cloud metadata
        ["10.218.97.11"],  # another private address, same subnet
        ["172.28.0.4"],  # another internal service
        [DEMO_WEB_IP, "127.0.0.1"],  # the bound address plus another one
        ["::ffff:10.218.97.10"],  # same IPv4 wrapped in IPv6 is not the bound address
    ],
)
def test_the_name_resolving_anywhere_else_is_refused(
    transport: RecordingTransport, answer: list[str]
) -> None:
    fetcher = Fetcher(
        FetchSettings(demo_hosts=DEMO), resolver=FakeResolver(answer), transport=transport
    )
    with pytest.raises(DisallowedDestinationError):
        _fetch(fetcher, PAGE)
    assert transport.requests == []


@pytest.mark.parametrize(
    "url",
    [
        "https://demo-web/",  # https: the exemption is http only
        "http://demo-web:443/",  # port 443 over http
        "https://demo-web:443/",
        "http://other-host/",  # another private name
        "http://demo-web.evil.example/",  # a suffix is not the host
        "http://10.218.97.10/",  # the address as a literal is never exempt
        "http://demo-web:8080/",  # ports outside 80/443 never pass
    ],
)
def test_the_exemption_covers_exactly_the_bound_origin(
    transport: RecordingTransport, url: str
) -> None:
    fetcher = Fetcher(
        FetchSettings(demo_hosts=DEMO), resolver=FakeResolver([DEMO_WEB_IP]), transport=transport
    )
    with pytest.raises(ToolError):
        _fetch(fetcher, url)
    assert transport.requests == []
