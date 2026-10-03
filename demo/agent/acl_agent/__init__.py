"""DataBot, the demo agent: an MCP + OpenAI-compatible client that knows only the gateway.

It runs inside the ``agent`` container (``edge`` network only), holds no credentials but the
bearer token an operator minted for it (``ACL_TOKEN``), and depends on httpx and the standard
library alone. ``client`` is also imported on the host by ``observability.smoke_traffic``.
"""
