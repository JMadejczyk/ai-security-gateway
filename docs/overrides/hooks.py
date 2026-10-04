"""MkDocs hook: links from docs pages to repository files outside docs/ point at GitHub."""

import re

REPO = "https://github.com/JMadejczyk/ai-security-gateway/blob/main/"
LINK = re.compile(r"\]\(\.\./([^)#\s]+)(#[^)\s]*)?\)")


def on_page_markdown(markdown: str, **_: object) -> str:
    return LINK.sub(lambda m: f"]({REPO}{m.group(1)}{m.group(2) or ''})", markdown)
