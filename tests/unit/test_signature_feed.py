"""The signature feed: strict schema, size cap, safe matching, sources, last-valid refresh."""

import asyncio
import contextlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
import yaml

from gateway.core.types import Channel
from gateway.feed.schema import (
    EMPTY_FEED,
    MAX_SIGNATURES,
    FeedError,
    FeedInvalidError,
    FeedUnavailableError,
    PatternType,
    SignatureFeed,
    glob_to_regex,
    parse_feed,
)
from gateway.feed.sources import FileFeedSource, HttpFeedSource, resolve_source
from gateway.feed.store import FeedRefresh, FeedStore
from gateway.policy.loader import PolicyLoader, PolicySnapshot
from gateway.telemetry import REGISTRY, FeedReloadResult

REPO_ROOT = Path(__file__).resolve().parents[2]
STARTER_FEED = REPO_ROOT / "feeds" / "signatures.json"
FEED_URL = "http://feed.test/signatures.json"


def signature(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": "inj.test",
        "source": "unit test",
        "pattern_type": "regex",
        "pattern": "(?i)ignore previous instructions",
        "severity": "high",
        "channels": ["llm", "mcp"],
    }
    entry.update(overrides)
    return entry


def feed_bytes(*signatures: dict[str, Any], version: str = "t.1", **extra: Any) -> bytes:
    return json.dumps({"version": version, "signatures": list(signatures), **extra}).encode()


def reloads(result: FeedReloadResult) -> float:
    return REGISTRY.get_sample_value("acl_feed_reloads_total", {"result": result.value}) or 0.0


# ------------------------------------------------------------------------------ schema


def test_the_starter_feed_is_valid():
    feed = parse_feed(STARTER_FEED.read_bytes())
    assert feed.version is not None
    assert 15 <= len(feed) <= 30
    kinds = {kind for kind in PatternType if feed.signatures(kind, Channel.MCP)}
    assert kinds == set(PatternType)


@pytest.mark.parametrize(
    ("data", "problem"),
    [
        pytest.param(feed_bytes(signature(), extra=1), "Extra inputs", id="unknown-root-field"),
        pytest.param(feed_bytes(signature(note="x")), "Extra inputs", id="unknown-entry-field"),
        pytest.param(json.dumps({"signatures": []}).encode(), "version", id="no-version"),
        pytest.param(feed_bytes(version="bad version"), "version", id="bad-version"),
        pytest.param(feed_bytes(signature(pattern_type="pickle")), "pattern_type", id="type"),
        pytest.param(feed_bytes(signature(severity="urgent")), "severity", id="severity"),
        pytest.param(feed_bytes(signature(channels=[])), "channels", id="no-channels"),
        pytest.param(feed_bytes(signature(channels=["smtp"])), "channels", id="bad-channel"),
        pytest.param(feed_bytes(signature(id="has space")), "id", id="bad-id"),
        pytest.param(feed_bytes(signature(), signature()), "appears twice", id="duplicate-id"),
        pytest.param(feed_bytes(signature(pattern="(open")), "invalid pattern", id="bad-regex"),
        pytest.param(feed_bytes(signature(pattern="x" * 1025)), "pattern", id="long-pattern"),
        pytest.param(
            feed_bytes(signature(pattern_type="path_glob", pattern="*/" * 9)),
            "wildcards",
            id="glob-wildcards",
        ),
        pytest.param(b"[1, 2]", "invalid feed", id="not-an-object"),
        pytest.param(b"{not json", "not valid UTF-8 JSON", id="not-json"),
        pytest.param(b"\xff\xfe", "not valid UTF-8 JSON", id="not-utf8"),
    ],
)
def test_invalid_feeds_are_rejected(data, problem):
    with pytest.raises(FeedInvalidError, match=problem):
        parse_feed(data)


def test_too_many_signatures_are_rejected():
    many = [signature(id=f"s{i}") for i in range(MAX_SIGNATURES + 1)]
    with pytest.raises(FeedInvalidError, match="signatures"):
        parse_feed(feed_bytes(*many), max_bytes=10 * 1024 * 1024)


def test_oversized_feed_is_rejected_before_parsing():
    data = feed_bytes(signature(source="x" * 150))
    with pytest.raises(FeedInvalidError, match="larger than"):
        parse_feed(data, max_bytes=len(data) - 1)
    assert parse_feed(data, max_bytes=len(data)).version == "t.1"


# --------------------------------------------------------------------------- matching


@pytest.mark.parametrize(
    ("glob", "path", "matches"),
    [
        ("~/.ssh/**", "~/.ssh/id_rsa", True),
        ("~/.ssh/**", "/home/u/.ssh/id_rsa", False),
        ("**/.ssh/**", "/home/u/.ssh/id_rsa", True),
        ("**/.env*", "/app/.env", True),
        ("**/.env*", "/app/.env.local", True),
        ("**/.env*", "config/.envrc", True),
        ("**/.env*", ".env", False),  # no leading segment: the bare-name glob covers it
        (".env*", ".env", True),
        ("**/etc/shadow", "/etc/shadow", True),
        ("**/etc/shadow", "../../etc/shadow", True),
        ("**/etc/shadow", "/etc/shadow.bak", False),
        ("**.pem", "certs/server.pem", True),
        ("*.pem", "certs/server.pem", False),  # one * stays within a segment
        ("*.pem", "server.pem", True),
        ("reports/?.md", "reports/a.md", False),  # ? is literal
        ("reports/?.md", "reports/?.md", True),
    ],
)
def test_path_globs(glob, path, matches):
    feed = parse_feed(
        feed_bytes(signature(pattern_type="path_glob", pattern=glob, channels=["mcp"]))
    )
    scan = feed.scan(Channel.MCP)
    scan.check(PatternType.PATH_GLOB, [path])
    assert bool(scan.result().matched) is matches
    assert glob_to_regex(glob)  # translatable on its own too


def test_signatures_apply_only_to_their_type_and_channels():
    feed = parse_feed(feed_bytes(signature(channels=["mcp"])))
    text = ["Ignore previous instructions"]
    llm = feed.scan(Channel.LLM)
    llm.check(PatternType.REGEX, text)
    assert llm.result().ids == ()
    mcp = feed.scan(Channel.MCP)
    mcp.check(PatternType.MCP_TOOL, text)  # a regex entry is not a tool signature
    mcp.check(PatternType.REGEX, text)
    assert mcp.result().ids == ("inj.test",)


def test_redos_pattern_times_out_safely():
    evil = signature(id="evil", pattern="(x+x+)+y")
    feed = parse_feed(feed_bytes(evil, signature(id="later", pattern="never-present")))
    scan = feed.scan(Channel.LLM)
    started = time.perf_counter()
    scan.check(PatternType.REGEX, ["x" * 5000])
    elapsed = time.perf_counter() - started
    result = scan.result()
    assert result.incomplete
    assert result.matched == ()
    assert elapsed < 1.0  # one pattern timeout (50 ms) plus the rest, far from catastrophic


def test_a_scan_has_an_overall_budget():
    slow = [signature(id=f"evil{i}", pattern="(x+x+)+y") for i in range(20)]
    scan = parse_feed(feed_bytes(*slow)).scan(Channel.LLM, budget_s=0.12)
    started = time.perf_counter()
    scan.check(PatternType.REGEX, ["x" * 5000])
    assert time.perf_counter() - started < 0.6  # 20 x 50 ms would be 1 s without the budget
    assert scan.result().incomplete


def test_empty_feed_matches_nothing():
    scan = EMPTY_FEED.scan(Channel.LLM)
    scan.check(PatternType.REGEX, ["Ignore previous instructions"])
    assert (EMPTY_FEED.version, scan.result().ids, scan.result().incomplete) == (None, (), False)


# ---------------------------------------------------------------------------- sources


def test_file_source_reads_at_most_one_byte_past_the_cap(tmp_path):
    path = tmp_path / "feed.json"
    path.write_bytes(b"x" * 100)
    assert FileFeedSource(path).fetch(10) == b"x" * 11
    with pytest.raises(FeedUnavailableError, match="cannot read"):
        FileFeedSource(tmp_path / "absent.json").fetch(10)


def test_relative_paths_resolve_against_the_policy_directory(tmp_path):
    source = resolve_source("feeds/signatures.json", base_dir=tmp_path)
    assert isinstance(source, FileFeedSource)
    assert source.path == tmp_path / "feeds" / "signatures.json"
    absolute = resolve_source("/etc/acl/feed.json", base_dir=tmp_path)
    assert isinstance(absolute, FileFeedSource)
    assert absolute.path == Path("/etc/acl/feed.json")
    assert isinstance(resolve_source(FEED_URL, base_dir=tmp_path), HttpFeedSource)
    with pytest.raises(FeedInvalidError, match="only http"):
        resolve_source("ftp://feed.test/signatures.json", base_dir=tmp_path)


@respx.mock(assert_all_called=False)
def test_http_source_fetches_a_feed(respx_mock):
    respx_mock.get(FEED_URL).mock(return_value=httpx.Response(200, content=feed_bytes()))
    assert parse_feed(HttpFeedSource(FEED_URL).fetch(1024)).version == "t.1"


@respx.mock(assert_all_called=False)
def test_http_source_caps_the_body(respx_mock):
    respx_mock.get(FEED_URL).mock(return_value=httpx.Response(200, content=b"x" * 5000))
    body = HttpFeedSource(FEED_URL).fetch(100)
    assert len(body) == 101
    with pytest.raises(FeedInvalidError, match="larger than"):
        parse_feed(body, max_bytes=100)


@respx.mock(assert_all_called=False)
def test_http_source_refuses_redirects(respx_mock):
    respx_mock.get(FEED_URL).mock(
        return_value=httpx.Response(302, headers={"location": "http://169.254.169.254/latest"})
    )
    internal = respx_mock.get("http://169.254.169.254/latest")
    with pytest.raises(FeedUnavailableError, match="redirects are refused"):
        HttpFeedSource(FEED_URL).fetch(1024)
    assert not internal.called


@pytest.mark.parametrize(
    ("response", "problem"),
    [
        (httpx.TimeoutException("slow"), "unreachable"),
        (httpx.ConnectError("refused"), "unreachable"),
        (httpx.Response(500), "HTTP 500"),
        (httpx.Response(200, headers={"content-encoding": "gzip"}, content=b"x"), "compressed"),
    ],
)
@respx.mock(assert_all_called=False)
def test_http_source_failures_are_unavailable(respx_mock, response, problem):
    route = respx_mock.get(FEED_URL)
    if isinstance(response, Exception):
        route.mock(side_effect=response)
    else:
        route.mock(return_value=response)
    with pytest.raises(FeedUnavailableError, match=problem):
        HttpFeedSource(FEED_URL, timeout_s=0.1).fetch(1024)


# ------------------------------------------------------------------------------ store


@pytest.fixture
def policy_dir(tmp_path) -> Path:
    shutil.copy(REPO_ROOT / "policy.yaml", tmp_path / "policy.yaml")
    shutil.copytree(REPO_ROOT / "feeds", tmp_path / "feeds")
    return tmp_path


def load_policy(policy_dir: Path, **signatures: Any) -> PolicySnapshot:
    path = policy_dir / "policy.yaml"
    document = yaml.safe_load(path.read_text())
    document["controls"]["signatures"].update(signatures)
    for key in [k for k, v in signatures.items() if v is None]:
        del document["controls"]["signatures"][key]
    path.write_text(yaml.safe_dump(document))
    return PolicyLoader().load(path)


def test_boot_loads_the_configured_feed(policy_dir):
    snapshot = load_policy(policy_dir)
    store = FeedStore.boot(lambda: snapshot)
    version = store.version
    assert version is not None
    assert version == json.loads(STARTER_FEED.read_text())["version"]
    assert REGISTRY.get_sample_value("acl_feed_info", {"version": version}) == 1


@pytest.mark.parametrize("breakage", ["missing", "invalid"])
def test_boot_refuses_a_broken_configured_feed(policy_dir, breakage):
    feed = policy_dir / "feeds" / "signatures.json"
    if breakage == "missing":
        feed.unlink()
    else:
        feed.write_text('{"version": "x", "signatures": [{"id": "bad"}]}')
    snapshot = load_policy(policy_dir)
    with pytest.raises(FeedError):
        FeedStore.boot(lambda: snapshot)


def test_boot_without_a_configured_feed_runs_with_none(policy_dir):
    snapshot = load_policy(policy_dir, feed=None)
    store = FeedStore.boot(lambda: snapshot)
    assert (store.version, len(store.current)) == (None, 0)


async def test_refresh_swaps_valid_feeds_and_keeps_the_last_valid_one(policy_dir):
    snapshot = load_policy(policy_dir)
    store = FeedStore.boot(lambda: snapshot)
    feed_file = policy_dir / "feeds" / "signatures.json"
    first = store.current

    assert (await store.refresh()).result is FeedReloadResult.UNCHANGED

    feed_file.write_bytes(feed_bytes(signature(), version="t.2"))
    outcome = await store.refresh()
    assert (outcome.result, outcome.version, store.version) == (FeedReloadResult.OK, "t.2", "t.2")
    assert store.current is not first

    for broken, result in (
        (b'{"version": "t.3", "signatures": [{"id": "x"}]}', FeedReloadResult.INVALID),
        (b"x" * (300 * 1024), FeedReloadResult.INVALID),
        (None, FeedReloadResult.UNAVAILABLE),
    ):
        before = reloads(result)
        if broken is None:
            feed_file.unlink()
        else:
            feed_file.write_bytes(broken)
        outcome = await store.refresh()
        assert (outcome.result, outcome.version) == (result, "t.2")
        assert outcome.error
        assert store.version == "t.2"
        assert reloads(result) == before + 1


async def test_refresh_follows_the_feed_named_by_the_current_policy(policy_dir):
    snapshots = [load_policy(policy_dir)]
    store = FeedStore.boot(lambda: snapshots[-1])
    (policy_dir / "other.json").write_bytes(feed_bytes(version="other.1"))
    snapshots.append(load_policy(policy_dir, feed="other.json"))
    assert (await store.refresh()).version == "other.1"


@respx.mock(assert_all_called=False)
async def test_refresh_from_http(policy_dir, respx_mock):
    respx_mock.get(FEED_URL).mock(return_value=httpx.Response(200, content=feed_bytes()))
    snapshot = load_policy(policy_dir, feed=FEED_URL)
    store = FeedStore.boot(lambda: snapshot)
    assert store.version == "t.1"
    respx_mock.get(FEED_URL).mock(side_effect=httpx.ConnectError("down"))
    assert (await store.refresh()).result is FeedReloadResult.UNAVAILABLE
    assert store.version == "t.1"


def test_feed_digest_tells_content_apart():
    a = parse_feed(feed_bytes(signature()))
    b = parse_feed(feed_bytes(signature(pattern="other")))
    assert a.digest != b.digest
    assert a.digest == parse_feed(feed_bytes(signature())).digest
    assert SignatureFeed().digest == EMPTY_FEED.digest


async def test_run_refreshes_every_refresh_s(policy_dir, monkeypatch):
    snapshot = load_policy(policy_dir, refresh_s=0.02)
    store = FeedStore.boot(lambda: snapshot)
    swapped = asyncio.Event()
    refresh = store.refresh

    async def observed_refresh() -> FeedRefresh:
        outcome = await refresh()
        if outcome.version == "bg.1":
            swapped.set()
        return outcome

    monkeypatch.setattr(store, "refresh", observed_refresh)
    task = asyncio.create_task(store.run())
    try:
        (policy_dir / "feeds" / "signatures.json").write_bytes(feed_bytes(version="bg.1"))
        await asyncio.wait_for(swapped.wait(), timeout=5)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
