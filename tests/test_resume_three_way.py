"""Approve, reject, and edit.

The third one is the point. A reviewer usually agrees with what the agent is
trying to do and disagrees with a detail of how — the wrong recipient, an
over-eager priority. Letting them fix the arguments in place is the feature
that makes an approval queue usable rather than annoying.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command
from pydantic import ValidationError
from sqlalchemy import select

from app import audit
from app.db import Ticket, get_sessionmaker
from app.graph.state import ResumeAction, ResumePayload
from tests.conftest import ai_tool_call


async def _pause_on_email(graph, script, thread, args=None):
    args = args or {"to": "ops@example.com", "subject": "Disk full", "body": "Node 3 at 98%."}
    script(
        ai_tool_call("send_email", args, "call-1", text="Emailing ops."),
        AIMessage(content="All done."),
    )
    config = thread()
    await graph.ainvoke({"messages": [HumanMessage(content="Email ops.")]}, config)
    return config, args


# --- approve -----------------------------------------------------------------


async def test_approve_runs_the_tool_with_the_original_args(graph, script, no_delivery, thread):
    config, args = await _pause_on_email(graph, script, thread)

    await graph.ainvoke(
        Command(resume={"action": "approve", "actor": "alice@example.com"}),
        config,
    )

    assert no_delivery.call_count == 1
    sent = no_delivery.calls[0]
    assert sent["To"] == args["to"]
    assert sent["Subject"] == args["subject"]

    entry = (await audit.for_thread(config["configurable"]["thread_id"]))[0]
    assert entry.decision == "approved"
    assert entry.decided_by == "alice@example.com"
    assert entry.args_modified is False
    assert entry.status == "executed"


# --- reject ------------------------------------------------------------------


async def test_reject_blocks_the_tool_and_tells_the_model_why(graph, script, no_delivery, thread):
    config, _ = await _pause_on_email(graph, script, thread)

    result = await graph.ainvoke(
        Command(
            resume={
                "action": "reject",
                "actor": "bob@example.com",
                "reason": "Ops already knows; don't spam them.",
            }
        ),
        config,
    )

    assert not no_delivery.called

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert "BLOCKED" in tool_messages[0].content
    assert "don't spam them" in tool_messages[0].content

    entry = (await audit.for_thread(config["configurable"]["thread_id"]))[0]
    assert entry.decision == "rejected"
    assert entry.decided_by == "bob@example.com"
    assert entry.status == "blocked"
    assert "spam" in entry.decision_reason


# --- edit --------------------------------------------------------------------


async def test_edit_sends_the_corrected_args_not_the_proposed_ones(
    graph, script, no_delivery, thread
):
    """The reviewer fixes the recipient before it goes out."""
    config, proposed = await _pause_on_email(
        graph,
        script,
        thread,
        args={"to": "everyone@example.com", "subject": "URGENT!!!", "body": "Node 3 at 98%."},
    )

    corrected = {
        "to": "oncall@example.com",
        "subject": "Disk usage on node 3",
        "body": "Node 3 at 98%.",
    }
    await graph.ainvoke(
        Command(
            resume={
                "action": "edit",
                "args": corrected,
                "actor": "alice@example.com",
                "reason": "Narrowed the recipient and toned down the subject.",
            }
        ),
        config,
    )

    assert no_delivery.call_count == 1
    sent = no_delivery.calls[0]
    assert sent["To"] == "oncall@example.com"
    assert sent["Subject"] == "Disk usage on node 3"

    entry = (await audit.for_thread(config["configurable"]["thread_id"]))[0]
    assert entry.decision == "edited"
    assert entry.args_modified is True
    # Both versions are kept: the audit trail shows what the agent wanted and
    # what the human let through.
    assert entry.proposed_args["to"] == "everyone@example.com"
    assert entry.final_args["to"] == "oncall@example.com"


async def test_edit_on_a_db_write_persists_the_edited_row(graph, script, thread):
    script(
        ai_tool_call(
            "create_ticket",
            {"title": "disk", "body": "node 3", "priority": "urgent"},
            "call-1",
        ),
        AIMessage(content="Filed."),
    )
    config = thread()
    await graph.ainvoke({"messages": [HumanMessage(content="File a ticket.")]}, config)

    await graph.ainvoke(
        Command(
            resume={
                "action": "edit",
                "args": {
                    "title": "Disk usage at 98% on node 3",
                    "body": "node 3",
                    "priority": "high",
                },
                "actor": "alice",
                "reason": "urgent is reserved for outages",
            }
        ),
        config,
    )

    async with get_sessionmaker()() as session:
        tickets = (await session.execute(select(Ticket))).scalars().all()

    assert len(tickets) == 1
    assert tickets[0].title == "Disk usage at 98% on node 3"
    assert tickets[0].priority == "high"


# --- payload validation ------------------------------------------------------


def test_edit_requires_args():
    with pytest.raises(ValidationError):
        ResumePayload(action=ResumeAction.EDIT)


def test_args_are_rejected_without_edit():
    with pytest.raises(ValidationError):
        ResumePayload(action=ResumeAction.APPROVE, args={"to": "x@example.com"})


def test_unknown_action_is_rejected():
    with pytest.raises(ValidationError):
        ResumePayload(action="maybe")


async def test_an_unreadable_resume_value_is_treated_as_a_rejection(
    graph, script, no_delivery, thread
):
    """When intent is unclear, the safe reading is 'do not act'."""
    config, _ = await _pause_on_email(graph, script, thread)

    result = await graph.ainvoke(Command(resume={"action": "approve", "junk": True}), config)

    assert not no_delivery.called
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert "BLOCKED" in tool_messages[0].content

    entry = (await audit.for_thread(config["configurable"]["thread_id"]))[0]
    assert entry.decision == "rejected"


async def test_a_bare_string_resume_still_works(graph, script, no_delivery, thread):
    """`Command(resume="approve")` is a convenience the CLI and notebooks use."""
    config, _ = await _pause_on_email(graph, script, thread)
    await graph.ainvoke(Command(resume="approve"), config)
    assert no_delivery.call_count == 1
