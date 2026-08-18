"""A guided walkthrough of the interlock, with nothing to install.

    python scripts/demo.py

No Docker, no Postgres, no API key, no SMTP server. The checkpointer is
MemorySaver, the database is a throwaway SQLite file, the model is scripted,
and SMTP delivery is replaced by a printer. Everything else — the policy
engine, the gate, the graph, the audit log — is the real code.

It walks one run through all three outcomes:

    1. the agent proposes a DB write   -> the graph pauses
    2. a human EDITS the arguments     -> the corrected row is written
    3. the agent proposes an email     -> the graph pauses again
    4. a human REJECTS it              -> nothing is sent
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Point the app at a throwaway database before anything reads settings.
_TMP = Path(tempfile.mkdtemp(prefix="interlock-demo-"))
os.environ["APP_DB_URL"] = f"sqlite+aiosqlite:///{(_TMP / 'demo.db').as_posix()}"
os.environ.setdefault("ANTHROPIC_API_KEY", "not-used-by-the-demo")

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.types import Command  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app import audit, llm  # noqa: E402
from app.db import Ticket, get_sessionmaker, init_db  # noqa: E402
from app.graph.build import memory_graph  # noqa: E402
from app.llm import ScriptedChatModel, tool_call_message  # noqa: E402

WIDTH = 78


def rule(title: str = "") -> None:
    if title:
        print(f"\n\033[1m{title}\033[0m")
        print("─" * WIDTH)
    else:
        print("─" * WIDTH)


def show_approval(payload: dict) -> None:
    print(f"\n  ⏸  THE GRAPH IS PAUSED. Nothing has happened yet.\n")
    print(f"     tool        {payload['tool']}")
    print(f"     effect      {payload['effect']}")
    print(f"     reversible  {'yes' if payload['reversible'] else 'no'}")
    print(f"     why gated   {payload['policy_reason']}")
    if payload.get("agent_rationale"):
        print(f"     agent says  {payload['agent_rationale']}")
    print("     proposed arguments:")
    for line in json.dumps(payload["args"], indent=2).splitlines():
        print(f"       {line}")


async def main() -> None:
    await init_db()

    # Stand in for SMTP so the demo needs no mail server. With Mailpit running
    # this would be a real send to a real (fake) inbox.
    sent: list = []

    async def fake_deliver(message):
        sent.append(message)
        print(f"     📧 delivered to {message['To']}: {message['Subject']}")

    import app.tools.email as email_tool

    email_tool._deliver = fake_deliver

    # The script the stand-in model replays, one message per turn.
    llm.set_llm(
        ScriptedChatModel(
            responses=[
                tool_call_message(
                    "create_ticket",
                    {
                        "title": "disk thing",
                        "body": "Node 3 root volume is at 98% capacity.",
                        "priority": "urgent",
                    },
                    "call-ticket",
                    text="I'll file a ticket for the disk alert first.",
                ),
                tool_call_message(
                    "send_email",
                    {
                        "to": "all-engineering@example.com",
                        "subject": "URGENT: DISK FULL!!!",
                        "body": "Node 3 is at 98%. Please advise.",
                    },
                    "call-email",
                    text="Now I'll notify the team by email.",
                ),
                AIMessage(
                    content=(
                        "I filed the ticket (with your corrections) but did not send the "
                        "email, since you rejected it. Let me know if you'd like a narrower "
                        "notification instead."
                    )
                ),
            ],
            calls=[],
        )
    )

    graph = memory_graph()
    config = {"configurable": {"thread_id": "demo-thread"}}

    rule("1. A user asks for something that touches real systems")
    request = "File a ticket for the disk alert on node 3 and email the team about it."
    print(f'\n  🧑 "{request}"')

    result = await graph.ainvoke({"messages": [HumanMessage(content=request)]}, config)

    # --- first gate: a database write -------------------------------------
    rule("2. The agent proposes a database write — the gate stops it")
    show_approval(result["__interrupt__"][0].value)

    async with get_sessionmaker()() as session:
        rows = (await session.execute(select(Ticket))).scalars().all()
    print(f"\n     tickets in the database right now: {len(rows)}  ← still nothing written")

    rule("3. The human EDITS the arguments rather than rejecting")
    print("\n  🧑 'Right idea, wrong details. Better title, and 'urgent' is for outages.'")
    result = await graph.ainvoke(
        Command(
            resume={
                "action": "edit",
                "args": {
                    "title": "Disk usage at 98% on node 3",
                    "body": "Node 3 root volume is at 98% capacity.",
                    "priority": "high",
                },
                "actor": "alice@example.com",
                "reason": "Clearer title; urgent is reserved for outages.",
            }
        ),
        config,
    )

    async with get_sessionmaker()() as session:
        rows = (await session.execute(select(Ticket))).scalars().all()
    print(f"\n     ✅ written: #{rows[0].id} [{rows[0].priority}] {rows[0].title}")
    print("        (the agent's version was never written)")

    # --- second gate: an email --------------------------------------------
    rule("4. The agent moves on to email — the gate stops it again")
    show_approval(result["__interrupt__"][0].value)

    rule("5. This time the human REJECTS")
    print("\n  🧑 'No. Do not mail all of engineering about a disk warning.'")
    result = await graph.ainvoke(
        Command(
            resume={
                "action": "reject",
                "actor": "alice@example.com",
                "reason": "Too broad an audience for a warning-level alert.",
            }
        ),
        config,
    )
    print(f"\n     emails actually sent: {len(sent)}  ← nothing left the system")
    print(f"\n  🤖 {result['messages'][-1].content}")

    # --- the audit trail ---------------------------------------------------
    rule("6. What the audit log recorded")
    entries = await audit.for_thread("demo-thread")
    for entry in entries:
        print(f"\n  • {entry.tool_name}  →  {entry.decision.upper()}  ({entry.status})")
        print(f"    decided by  {entry.decided_by}")
        print(f"    reason      {entry.decision_reason}")
        if entry.args_modified:
            print(f"    proposed    {json.dumps(entry.proposed_args)}")
            print(f"    ran with    {json.dumps(entry.final_args)}")
        else:
            print(f"    args        {json.dumps(entry.final_args)}")

    rule()
    print(
        "\nThat is the whole idea: the agent decided what to do, a human decided\n"
        "whether it happened, and the table above can prove who chose what.\n"
    )


if __name__ == "__main__":
    asyncio.run(main())
