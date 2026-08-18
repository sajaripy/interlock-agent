"""Several tools in one turn, and the crash-replay guard.

Claude can request more than one tool in a single turn, and the tiers can be
mixed. Each call is gated independently, and every one must end up with a
ToolMessage — the API rejects a turn where a tool call has no result, so a
rejection has to produce one too.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from app import audit
from app.db import Ticket, get_sessionmaker


def multi_tool_message(*calls: tuple[str, dict, str], text: str = "") -> AIMessage:
    return AIMessage(
        content=text,
        tool_calls=[
            {"name": name, "args": args, "id": call_id, "type": "tool_call"}
            for name, args, call_id in calls
        ],
    )


async def test_auto_tool_runs_while_the_risky_one_waits(graph, script, no_delivery, thread, monkeypatch):
    """A mixed turn: the read-only call goes through, the email stops."""
    ran: list[str] = []

    async def run_tool(name: str, args: dict) -> str:
        ran.append(name)
        return "HTTP 200 OK\n\nstatus: degraded"

    monkeypatch.setattr("app.graph.nodes.run_tool", run_tool)

    script(
        multi_tool_message(
            ("fetch_url", {"url": "https://example.com/status"}, "call-fetch"),
            ("send_email", {"to": "ops@example.com", "subject": "S", "body": "B"}, "call-email"),
            text="Checking status, then emailing ops.",
        ),
        AIMessage(content="Done."),
    )
    config = thread()

    result = await graph.ainvoke({"messages": [HumanMessage(content="Check and notify.")]}, config)

    assert "__interrupt__" in result
    assert ran == ["fetch_url"], "the auto-tier call should have run on its own"
    assert not no_delivery.called, "the gated call must still be waiting"
    assert result["__interrupt__"][0].value["tool"] == "send_email"


async def test_every_tool_call_in_a_turn_gets_a_result(graph, script, no_delivery, thread):
    """Two gated calls, approved and rejected. Both need a ToolMessage."""
    script(
        multi_tool_message(
            ("create_ticket", {"title": "Disk at 98% on node 3"}, "call-ticket"),
            ("send_email", {"to": "all@example.com", "subject": "S", "body": "B"}, "call-email"),
        ),
        AIMessage(content="Ticket filed; email not sent."),
    )
    config = thread()

    await graph.ainvoke({"messages": [HumanMessage(content="File and notify.")]}, config)

    # First gate: the ticket.
    result = await graph.ainvoke(
        Command(resume={"action": "approve", "actor": "alice"}), config
    )
    assert "__interrupt__" in result, "the second gated call should pause too"
    assert result["__interrupt__"][0].value["tool"] == "send_email"

    # Second gate: the email.
    result = await graph.ainvoke(
        Command(resume={"action": "reject", "actor": "alice", "reason": "Too broad."}), config
    )

    assert "__interrupt__" not in result
    assert not no_delivery.called

    resolved = {m.tool_call_id for m in result["messages"] if isinstance(m, ToolMessage)}
    assert resolved == {"call-ticket", "call-email"}

    async with get_sessionmaker()() as session:
        from sqlalchemy import select

        tickets = (await session.execute(select(Ticket))).scalars().all()
    assert len(tickets) == 1

    entries = await audit.for_thread(config["configurable"]["thread_id"])
    assert {e.tool_name: e.decision for e in entries} == {
        "create_ticket": "approved",
        "send_email": "rejected",
    }


# --- the crash-replay guard --------------------------------------------------


async def test_a_replayed_execute_does_not_send_twice(graph, script, no_delivery, thread):
    """If the process dies after the tool runs but before the checkpoint is
    written, LangGraph re-runs the node. The audit table is what stops the
    second send."""
    script(
        multi_tool_message(("send_email", {"to": "a@example.com", "subject": "S", "body": "B"}, "call-1")),
        AIMessage(content="Sent."),
    )
    config = thread()
    thread_id = config["configurable"]["thread_id"]

    await graph.ainvoke({"messages": [HumanMessage(content="Email a.")]}, config)
    await graph.ainvoke(Command(resume={"action": "approve", "actor": "alice"}), config)
    assert no_delivery.call_count == 1

    # Simulate the replay: run the executor again against the same state.
    from app.graph import nodes

    pending = {
        "tool_call_id": "call-1",
        "name": "send_email",
        "proposed_args": {"to": "a@example.com", "subject": "S", "body": "B"},
        "final_args": {"to": "a@example.com", "subject": "S", "body": "B"},
        "mode": "approve",
        "policy_source": "policy:tools.send_email",
        "policy_reason": "test",
        "floor_applied": False,
        "effect": "sends email",
        "reversible": False,
        "requested_at": audit.utcnow().isoformat(),
    }
    replay = await nodes.execute(
        {"messages": [], "pending": pending, "review": None},
        {"configurable": {"thread_id": thread_id}},
    )

    assert no_delivery.call_count == 1, "the replay sent the email a second time"
    assert "already completed" in replay["messages"][0].content

    entries = await audit.for_thread(thread_id)
    assert [e.status for e in entries] == ["executed", "skipped_duplicate"]


async def test_a_failing_tool_is_recorded_and_reported_to_the_model(
    graph, script, thread, monkeypatch
):
    async def boom(name: str, args: dict) -> str:
        raise ConnectionRefusedError("no SMTP server on localhost:1025")

    monkeypatch.setattr("app.graph.nodes.run_tool", boom)

    script(
        multi_tool_message(("send_email", {"to": "a@example.com", "subject": "S", "body": "B"}, "call-1")),
        AIMessage(content="The email failed to send."),
    )
    config = thread()

    await graph.ainvoke({"messages": [HumanMessage(content="Email a.")]}, config)
    result = await graph.ainvoke(Command(resume={"action": "approve", "actor": "alice"}), config)

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert "ERROR" in tool_messages[0].content

    entry = (await audit.for_thread(config["configurable"]["thread_id"]))[0]
    assert entry.status == "failed"
    assert "ConnectionRefusedError" in entry.error
    assert entry.decision == "approved", "who approved it still matters when it fails"
