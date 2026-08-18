"""Risk tier 2: a database write. Approval required.

Reversible, but it changes state that other systems read. A human sees the
proposed row before it lands, and can correct the title or priority first —
which is the common real-world case, not a rejection.
"""

from __future__ import annotations

from langchain_core.tools import tool

from app.db import Ticket, get_sessionmaker

VALID_PRIORITIES = ("low", "normal", "high", "urgent")


@tool
async def create_ticket(
    title: str,
    body: str = "",
    priority: str = "normal",
    created_by: str = "agent",
) -> str:
    """Create a support ticket in the database.

    Writes a real row. Use it when the user asks to log, file, raise, or open
    a ticket or issue.

    Args:
        title: Short one-line summary of the ticket.
        body: Full description of the problem.
        priority: One of low, normal, high, urgent.
        created_by: Who this ticket is filed on behalf of.
    """
    if not title.strip():
        return "ERROR: title must not be empty."
    if priority not in VALID_PRIORITIES:
        return f"ERROR: priority must be one of {', '.join(VALID_PRIORITIES)}; got '{priority}'."

    ticket = Ticket(
        title=title.strip(),
        body=body,
        priority=priority,
        created_by=created_by,
    )
    async with get_sessionmaker()() as session:
        session.add(ticket)
        await session.commit()
        ticket_id = ticket.id

    return f"Created ticket #{ticket_id} ({priority}): {ticket.title}"
