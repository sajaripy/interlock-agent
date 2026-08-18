"""Risk tier 3: leaves the system. Always requires approval.

The message is genuinely sent over SMTP — to Mailpit, which accepts everything
and delivers nothing. So the code path is real (a bug here is a real bug) while
the blast radius is zero.

`_deliver` is deliberately a separate module-level function: the tests
monkeypatch it and assert it was *not* called while the graph was paused. That
assertion is the whole point of the project, so it needs a seam to hang on.
"""

from __future__ import annotations

from email.message import EmailMessage

import aiosmtplib
from langchain_core.tools import tool

from app.config import get_settings


async def _deliver(message: EmailMessage) -> None:
    """The actual, irreversible act. Nothing above this line leaves the box."""
    settings = get_settings()
    await aiosmtplib.send(
        message,
        hostname=settings.smtp_host,
        port=settings.smtp_port,
        start_tls=False,
    )


@tool
async def send_email(to: str, subject: str, body: str) -> str:
    """Send an email.

    This actually delivers a message to an external recipient and cannot be
    undone. Use it only when the user explicitly asks to email someone.

    Args:
        to: Recipient email address.
        subject: Subject line.
        body: Plain-text body of the email.
    """
    if "@" not in to:
        return f"ERROR: '{to}' is not a valid email address."
    if not subject.strip():
        return "ERROR: subject must not be empty."

    settings = get_settings()
    message = EmailMessage()
    message["From"] = settings.email_from
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)

    await _deliver(message)
    return f"Email sent to {to} with subject '{subject}'."
