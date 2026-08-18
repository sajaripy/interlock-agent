"""Graph assembly and checkpointer wiring.

The checkpointer is the reason any of this works. `interrupt()` stops the graph
mid-run; the checkpoint is what lets that paused run outlive the request that
started it — and the process that served it. A pause that evaporates on restart
would defeat the point, which is why Postgres is the default and MemorySaver is
confined to tests.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.config import get_settings
from app.graph import nodes
from app.graph.state import AgentState


def build_graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("agent", nodes.agent)
    builder.add_node("policy_gate", nodes.policy_gate)
    builder.add_node("human_review", nodes.human_review)
    builder.add_node("execute", nodes.execute)
    builder.add_node("rejected", nodes.rejected)
    builder.add_node("denied", nodes.denied)

    builder.add_edge(START, "agent")

    # The model either asked for a tool or it is finished.
    builder.add_conditional_edges(
        "agent",
        nodes.route_from_agent,
        {"policy_gate": "policy_gate", "__end__": END},
    )

    # The gate: auto runs, approve pauses, deny blocks. When nothing is
    # pending, every tool call from this turn is resolved and the model
    # gets to see the results.
    builder.add_conditional_edges(
        "policy_gate",
        nodes.route_from_gate,
        {
            "execute": "execute",
            "human_review": "human_review",
            "denied": "denied",
            "agent": "agent",
        },
    )

    builder.add_conditional_edges(
        "human_review",
        nodes.route_from_review,
        {"execute": "execute", "rejected": "rejected"},
    )

    # Each resolved call goes back to the gate, which picks up the next one
    # from the same turn or hands control back to the model.
    builder.add_edge("execute", "policy_gate")
    builder.add_edge("rejected", "policy_gate")
    builder.add_edge("denied", "policy_gate")

    return builder.compile(checkpointer=checkpointer)


@asynccontextmanager
async def postgres_graph() -> AsyncIterator[CompiledStateGraph]:
    """Production wiring: a Postgres-backed graph, tables created on entry."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    dsn = get_settings().checkpoint_dsn
    async with AsyncPostgresSaver.from_conn_string(dsn) as checkpointer:
        await checkpointer.setup()
        yield build_graph(checkpointer)


def memory_graph() -> CompiledStateGraph:
    """Test wiring only. A pause that cannot survive a restart is not an
    approval workflow, so this must never be the production path."""
    from langgraph.checkpoint.memory import MemorySaver

    return build_graph(MemorySaver())
