"""The HTTP surface.

Three endpoints carry the whole workflow:

    POST /runs                      start a run; it returns either an answer
                                    or an approval request
    GET  /runs/{thread_id}          look at a parked run
    POST /runs/{thread_id}/resume   approve, reject, or edit

`thread_id` is not an application concept layered on top of LangGraph — it *is*
the checkpoint thread. That is what lets a run be started by one process,
reviewed an hour later, and resumed by a different one.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app import audit
from app.api.schemas import (
    ApprovalRequest,
    AuditEntryOut,
    MessageOut,
    PolicyOut,
    PolicyRuleOut,
    ResumeRequest,
    RunState,
    StartRunRequest,
)
from app.config import get_settings
from app.db import dispose_db, init_db
from app.graph.build import build_graph
from app.observability import get_callbacks
from app.policy import Mode, get_policy_engine, stricter
from app.tools import TOOLS, floor_for

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open the Postgres checkpointer for the life of the process.

    The saver holds a connection pool, so it is created once here rather than
    per request. `setup()` creates LangGraph's own tables if they are missing.
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    settings = get_settings()
    await init_db()

    async with AsyncPostgresSaver.from_conn_string(settings.checkpoint_dsn) as checkpointer:
        await checkpointer.setup()
        app.state.graph = build_graph(checkpointer)
        app.state.checkpointer = checkpointer
        logger.info("Interlock ready. Model=%s", settings.model)
        try:
            yield
        finally:
            await dispose_db()


app = FastAPI(
    title="Interlock",
    version="0.1.0",
    description=(
        "An agent that can act, but stops for a human before anything risky. "
        "Runs are LangGraph threads checkpointed in Postgres, so a pending "
        "approval survives a restart."
    ),
    lifespan=lifespan,
)


def get_graph(request: Request) -> CompiledStateGraph:
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(status_code=503, detail="Graph is not ready.")
    return graph


def _run_config(thread_id: str) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": thread_id},
        "callbacks": get_callbacks(),
        "recursion_limit": 50,
    }


# --- serialization -----------------------------------------------------------


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part)
    return str(content)


def _message_out(message: Any) -> MessageOut | None:
    if isinstance(message, HumanMessage):
        return MessageOut(role="human", content=_text(message.content))
    if isinstance(message, AIMessage):
        return MessageOut(
            role="ai",
            content=_text(message.content),
            tool_calls=[dict(call) for call in (message.tool_calls or [])],
        )
    if isinstance(message, ToolMessage):
        return MessageOut(
            role="tool",
            content=_text(message.content),
            tool_call_id=message.tool_call_id,
        )
    if isinstance(message, SystemMessage):
        return MessageOut(role="system", content=_text(message.content))
    return None


def _final_reply(messages: list[Any]) -> str | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            text = _text(message.content).strip()
            if text:
                return text
    return None


def _state_from_values(
    thread_id: str, values: dict[str, Any], approval: dict[str, Any] | None
) -> RunState:
    messages = values.get("messages", []) or []
    return RunState(
        thread_id=thread_id,
        status="awaiting_approval" if approval else "completed",
        approval=ApprovalRequest.model_validate(approval) if approval else None,
        reply=None if approval else _final_reply(messages),
        messages=[out for out in (_message_out(m) for m in messages) if out is not None],
    )


def _approval_from_result(result: dict[str, Any]) -> dict[str, Any] | None:
    """Pull the pending approval out of an ainvoke result, if the run paused."""
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    value = interrupts[0].value
    return value if isinstance(value, dict) else {"type": "approval_request", "value": value}


# --- endpoints ---------------------------------------------------------------


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/runs", response_model=RunState, tags=["runs"], status_code=201)
async def start_run(
    body: StartRunRequest,
    graph: CompiledStateGraph = Depends(get_graph),
) -> RunState:
    """Start a run.

    Returns as soon as the agent either finishes or hits a gated tool call. A
    `status` of `awaiting_approval` means the graph is parked in Postgres and
    nothing further will happen until someone calls the resume endpoint.
    """
    thread_id = body.thread_id or f"run-{uuid.uuid4().hex[:12]}"
    config = _run_config(thread_id)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=body.message)]},
        config,
    )
    return _state_from_values(thread_id, result, _approval_from_result(result))


@app.get("/runs/{thread_id}", response_model=RunState, tags=["runs"])
async def get_run(
    thread_id: str = Path(..., description="The LangGraph checkpoint thread."),
    graph: CompiledStateGraph = Depends(get_graph),
) -> RunState:
    """Inspect a run — including exactly what it is waiting for."""
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    if not snapshot.created_at:
        raise HTTPException(status_code=404, detail=f"No run with thread_id '{thread_id}'.")

    approval = None
    if snapshot.interrupts:
        value = snapshot.interrupts[0].value
        approval = value if isinstance(value, dict) else None

    return _state_from_values(thread_id, snapshot.values, approval)


@app.post("/runs/{thread_id}/resume", response_model=RunState, tags=["runs"])
async def resume_run(
    body: ResumeRequest,
    thread_id: str = Path(...),
    graph: CompiledStateGraph = Depends(get_graph),
) -> RunState:
    """Approve, reject, or edit the pending call, then let the run continue.

    `edit` is the interesting one: `args` replaces what the agent proposed, and
    both versions are kept in the audit log.
    """
    config = _run_config(thread_id)

    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    if not snapshot.created_at:
        raise HTTPException(status_code=404, detail=f"No run with thread_id '{thread_id}'.")
    if not snapshot.interrupts:
        raise HTTPException(
            status_code=409,
            detail=f"Run '{thread_id}' is not waiting for a decision.",
        )

    result = await graph.ainvoke(Command(resume=body.model_dump(mode="json")), config)
    return _state_from_values(thread_id, result, _approval_from_result(result))


# --- the audit trail ---------------------------------------------------------


@app.get("/runs/{thread_id}/audit", response_model=list[AuditEntryOut], tags=["audit"])
async def thread_audit(thread_id: str = Path(...)) -> list[AuditEntryOut]:
    """Every gated decision made during one run."""
    entries = await audit.for_thread(thread_id)
    return [AuditEntryOut.model_validate(audit.to_dict(e)) for e in entries]


@app.get("/audit", response_model=list[AuditEntryOut], tags=["audit"])
async def recent_audit(limit: int = Query(100, ge=1, le=1000)) -> list[AuditEntryOut]:
    """The most recent decisions across all runs."""
    entries = await audit.recent(limit)
    return [AuditEntryOut.model_validate(audit.to_dict(e)) for e in entries]


# --- introspection -----------------------------------------------------------


@app.get("/policy", response_model=PolicyOut, tags=["meta"])
async def current_policy() -> PolicyOut:
    """What the gate will do with each tool right now, floors included."""
    config = get_policy_engine().config
    rules = []
    for name, spec in TOOLS.items():
        rule = config.tools.get(name)
        configured = rule.mode if rule else config.default
        floor = floor_for(name)
        rules.append(
            PolicyRuleOut(
                tool=name,
                configured_mode=configured.value,
                floor=floor.value,
                effective_mode=stricter(configured, floor).value,
                reason=(rule.reason if rule else "not listed; using default"),
                effect=spec.effect,
                reversible=spec.reversible,
            )
        )
    return PolicyOut(
        default=config.default.value,
        hot_reload=config.hot_reload,
        rules=sorted(rules, key=lambda r: Mode(r.effective_mode) != Mode.AUTO),
    )
