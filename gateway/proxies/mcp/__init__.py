"""MCP entry point: an MCP server to the agent, an MCP client to each upstream server.

- `wire`: JSON-RPC 2.0 envelopes and the MCP ``2025-06-18`` tools-only subset.
- `upstream`: `MCPUpstream`, one upstream session per downstream session (hand-rolled client).
- `sessions`: `MCPSessionRegistry`, downstream session ids bound to their gateway session.
- `pins`: operator-pinned tool schemas (``pins/<server>.json``).
- `downstream`: `MCPProxy`, the agent-facing ``/mcp/{server}`` endpoint.
"""
