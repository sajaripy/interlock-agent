"""Risk tier 1: read-only. Auto-approved.

A GET has no side effect and is cheap to retry, so it runs without a human.
It is still policy-evaluated and still audited — "auto" means nobody is
interrupted, not that nobody is watching.
"""

from __future__ import annotations

from urllib.parse import urlparse

import httpx
from langchain_core.tools import tool

from app.config import get_settings


@tool
async def fetch_url(url: str) -> str:
    """Fetch a URL with an HTTP GET and return its status and body.

    Use this to read public web pages or JSON APIs. Read-only: it cannot
    change anything. The body is truncated if the response is large.

    Args:
        url: An absolute http:// or https:// URL.
    """
    settings = get_settings()

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"ERROR: refusing to fetch non-HTTP URL scheme '{parsed.scheme}'."
    if not parsed.netloc:
        return "ERROR: url must be absolute, e.g. https://example.com/path."

    async with httpx.AsyncClient(
        timeout=settings.http_timeout_seconds,
        follow_redirects=True,
    ) as client:
        response = await client.get(url)

    body = response.text
    truncated = ""
    if len(body) > settings.http_max_bytes:
        body = body[: settings.http_max_bytes]
        truncated = f"\n\n[truncated at {settings.http_max_bytes} characters]"

    return f"HTTP {response.status_code} {response.reason_phrase}\n\n{body}{truncated}"
