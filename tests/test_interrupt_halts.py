"""The load-bearing tests: the graph stops *before* the side effect.

If these pass, the interlock works. If any of them fail, nothing else in the
project matters.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from app import audit
from tests.conftest import ai_tool_call


async def test_email_is_not_sent_while_awaiting_approval(graph, script, no_delivery, thread):
    """The whole project in one assertion: paused, and nothing went out."""
    script(
        ai_tool_call(
            "send_email",
            {"to": "ops@example.com", "subject": "Disk full", "body": "Node 3 is at 98%."},
            "call-1",
            text="I'll email the ops team about the disk alert.",
        )
    )
    config = thread()

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Email ops about the disk alert.")]},
        config,
    )

    assert "__interrupt__" in result, "graph should have paused for approval"
    assert not no_delivery.called, "email was delivered before anyone approved it"

    payload = result["__interrupt__"][0].value
    assert payload["type"] == "approval_request"
    assert payload["tool"] == "send_email"
    assert payload["args"]["to"] == "ops@example.com"
    assert payload["reversible"] is False


async def test_db_write_also_pauses(graph, script, thread):
    from app.db import Ticket, get_sessionmaker
    from sqlalchemy import select

    script(ai_tool_call("create_ticket", {"title": "Disk full on node 3"}, "call-1"))
    config = thread()

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="File a ticket for the disk alert.")]},
        config,
    )

    assert "__interrupt__" in result
    async with get_sessionmaker()() as session:
        rows = (await session.execute(select(Ticket))).scalars().all()
    assert rows == [], "a ticket was written before approval"


async def test_read_only_tool_runs_without_pausing(graph, script, thread, monkeypatch):
    """`auto` means no human is interrupted — but it is still audited."""
    calls: list[str] = []

    async def run_tool(name: str, args: dict) -> str:
        calls.append(name)
        return "HTTP 200 OK\n\nstatus: green"

    monkeypatch.setattr("app.graph.nodes.run_tool", run_tool)

    script(
        ai_tool_call("fetch_url", {"url": "https://example.com/status"}, "call-1"),
        AIMessage(content="The status page reports green."),
    )
    config = thread()

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="What does the status page say?")]},
        config,
    )

    assert "__interrupt__" not in result, "a read-only call should not stop the graph"
    assert calls == ["fetch_url"]
    assert result["messages"][-1].content == "The status page reports green."

    entries = await audit.for_thread(config["configurable"]["thread_id"])
    assert [e.decision for e in entries] == ["auto_approved"]
    assert entries[0].status == "executed"


async def test_pause_survives_a_new_graph_object(graph, script, no_delivery, thread):
    """Resume does not depend on the object that started the run.

    With MemorySaver the store is in-process, so this only proves the resume
    path reads from the checkpoint rather than from local variables. The
    Postgres saver is what extends that across a restart.
    """
    script(
        ai_tool_call("send_email", {"to": "a@example.com", "subject": "Hi", "body": "Hello"}, "call-1")
    )
    config = thread()

    await graph.ainvoke({"messages": [HumanMessage(content="Email a@example.com.")]}, config)
    assert not no_delivery.called

    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("human_review",), "run should be parked at the approval node"
    assert snapshot.interrupts, "the checkpoint should carry the pending interrupt"

    resumed = await graph.ainvoke(
        Command(resume={"action": "approve", "actor": "alice@example.com"}),
        config,
    )

    assert no_delivery.call_count == 1
    assert "__interrupt__" not in resumed


async def test_side_effect_fires_exactly_once_across_the_pause(graph, script, no_delivery, thread):
    """Guards the replay trap.

    On resume LangGraph re-runs `human_review` from its first line. If any side
    effect lived above the interrupt() call, it would run twice. Approval is
    isolated in its own node precisely so this count stays at 1.
    """
    script(
        ai_tool_call("send_email", {"to": "b@example.com", "subject": "Once", "body": "Only once"}, "call-1")
    )
    config = thread()

    await graph.ainvoke({"messages": [HumanMessage(content="Email b@example.com.")]}, config)
    await graph.ainvoke(Command(resume={"action": "approve", "actor": "alice"}), config)

    assert no_delivery.call_count == 1, "the side effect fired more than once across the pause"

    entries = await audit.for_thread(config["configurable"]["thread_id"])
    assert len(entries) == 1, "the pause should not duplicate audit rows either"


async def test_denied_tool_never_reaches_the_human_or_the_tool(graph, script, thread, monkeypatch):
    """A default-denied tool is blocked without interrupting anybody."""
    from app.policy import Mode, PolicyConfig, PolicyEngine, set_policy_engine

    # A policy where send_email is explicitly forbidden.
    set_policy_engine(
        PolicyEngine(config=PolicyConfig(default=Mode.DENY, hot_reload=False, tools={}))
    )

    ran: list[str] = []

    async def run_tool(name: str, args: dict) -> str:
        ran.append(name)
        return "should never happen"

    monkeypatch.setattr("app.graph.nodes.run_tool", run_tool)

    script(
        ai_tool_call("send_email", {"to": "c@example.com", "subject": "X", "body": "Y"}, "call-1"),
        AIMessage(content="I'm not allowed to send email."),
    )
    config = thread()

    result = await graph.ainvoke({"messages": [HumanMessage(content="Email c@example.com.")]}, config)

    assert "__interrupt__" not in result, "a denied tool must not bother a human"
    assert ran == [], "a denied tool must not run"

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert "BLOCKED BY POLICY" in tool_messages[0].content

    entries = await audit.for_thread(config["configurable"]["thread_id"])
    assert entries[0].decision == "denied"
    assert entries[0].status == "blocked"
