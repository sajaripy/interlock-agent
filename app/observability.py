"""Optional Langfuse tracing.

Entirely opt-in: with no Langfuse keys set, this returns no callbacks and the
app behaves identically. Tracing shows you the model's reasoning and token
spend; the `audit_log` table shows you who authorised what. They answer
different questions and you want both.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)

_callbacks: list[Any] | None = None


def get_callbacks() -> list[Any]:
    """LangChain callbacks to attach to every graph invocation."""
    global _callbacks
    if _callbacks is not None:
        return _callbacks

    settings = get_settings()
    if not settings.langfuse_enabled:
        _callbacks = []
        return _callbacks

    try:
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key or "")
        os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key or "")
        os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)
        from langfuse.langchain import CallbackHandler

        _callbacks = [CallbackHandler()]
        logger.info("Langfuse tracing enabled (%s).", settings.langfuse_host)
    except Exception as exc:
        # Never let the tracer take down the agent.
        logger.warning("Langfuse is configured but could not start: %s", exc)
        _callbacks = []

    return _callbacks
