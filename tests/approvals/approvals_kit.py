"""Helpers for the approval suites (imported by name: tests/approvals is not a package)."""

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

from gateway_testkit import MutableClock

from gateway.approvals.store import ApprovalStore


@dataclass
class StoreUnderTest:
    kind: str
    store: ApprovalStore
    clock: MutableClock


# The MCP suites' harness (real in-process MCP servers, a minimal MCP client) lives in
# tests/mcp, which is not on the import path of this directory; loaded by name so both pytest
# and pyright are happy (pyright sees it as Any).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcp"))
mcp_harness = importlib.import_module("mcp_harness")
upstreams = importlib.import_module("upstreams")
pin_kit = importlib.import_module("pin_kit")
