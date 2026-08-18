"""The graph nodes.

Read `human_review` first — the rest exists to serve it.

The graph is a ReAct loop with a gate spliced in between "the model asked for a
tool" and "the tool runs":

    agent -> policy_gate -> [ execute | human_review | denied ]
                                          |
                                    [ execute | rejected ]

Nodes are kept deliberately small and single-purpose. In particular, the node
that waits for a human does nothing except wait and report the answer; the node
that performs side effects never waits. That separation is what makes the
resume path safe.
"""

from __future__ import annotations

import datetime as dt
import time
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from app import audit
from app.graph.state import AgentState, PendingCall, ResumeAction, ResumePayload, Review
from app.llm import get_llm
from app.policy import Mode, get_policy_engine
from app.tools import all_tools, floor_for, get_spec, run_tool

SYSTEM_PROMPT = """You are Interlock, an operations assistant that can take real actions.

You have three tools:
  - fetch_url: reads a web page or JSON API. Read-only.
  - create_ticket: writes a support ticket to the database.
  - send_email: sends a real email to a real person.

Some of these are gated: a human reviews the call before it runs, and may edit
your arguments or reject the call outright. This is expected and normal. When a
tool result says a call was rejected or blocked, do not retry the same call.
Explain the outcome to the user and, if it helps, propose a different approach.

Be concrete. When you propose a gated action, state plainly in your message what
you are about to do and why, because a human is going to read it before deciding.
"""


def _utcnow_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _last_ai_message(messages: list[Any]) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def next_pending_call(messages: list[Any]) -> dict[str, Any] | None:
    """The first tool call from the latest AI turn that has no result yet.

    Claude can request several tools in one turn. Each one is gated separately,
    so we resolve them one at a time and track progress by which tool_call_ids
    already have a matching ToolMessage. Every call must end up with a result —
    the API rejects a turn where one is missing — so rejections and denials
    still append a ToolMessage explaining themselves.
    """
    latest = _last_ai_message(messages)
    if latest is None or not latest.tool_calls:
        return None

    resolved = {
        message.tool_call_id
        for message in messages
        if isinstance(message, ToolMessage)
    }
    for call in latest.tool_calls:
        if call["id"] not in resolved:
            return call
    return None


def _thread_id(config: RunnableConfig) -> str:
    return str((config.get("configurable") or {}).get("thread_id", "unknown"))


# --- nodes -------------------------------------------------------------------


async def agent(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Ask the model what to do next."""
    model = get_llm().bind_tools(all_tools())
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    response = await model.ainvoke(messages, config)
    return {"messages": [response]}


async def policy_gate(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Evaluate the next unresolved tool call against the policy.

    Pure decision-making: it classifies the call and records the verdict in
    state. It never runs anything.
    """
    call = next_pending_call(state["messages"])
    if call is None:
        return {"pending": None, "review": None}

    name = call["name"]
    decision = get_policy_engine().evaluate(name, floor=floor_for(name))
    spec = get_spec(name)
    args = dict(call.get("args") or {})

    pending: PendingCall = {
        "tool_call_id": call["id"],
        "name": name,
        "proposed_args": args,
        "final_args": dict(args),
        "mode": decision.mode.value,
        "policy_source": decision.source,
        "policy_reason": decision.reason,
        "floor_applied": decision.floor_applied,
        "effect": spec.effect if spec else "Unknown tool.",
        "reversible": spec.reversible if spec else False,
        "requested_at": _utcnow_iso(),
    }
    return {"pending": pending, "review": None}


async def human_review(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Pause the graph and wait for a person.

    This node's only job is to turn a human's answer into a value in state. It
    performs no side effects at all, on purpose — see the comment below.
    """
    pending = state["pending"]
    assert pending is not None, "human_review reached with no pending call"

    payload = {
        "type": "approval_request",
        "tool_call_id": pending["tool_call_id"],
        "tool": pending["name"],
        "args": pending["proposed_args"],
        "mode": pending["mode"],
        "policy_source": pending["policy_source"],
        "policy_reason": pending["policy_reason"],
        "floor_applied": pending["floor_applied"],
        "effect": pending["effect"],
        "reversible": pending["reversible"],
        "requested_at": pending["requested_at"],
        "agent_rationale": _rationale(state),
        "actions": ["approve", "reject", "edit"],
    }

    # ------------------------------------------------------------------------
    # THE ONE RULE OF THIS NODE.
    #
    # When the graph resumes, LangGraph re-runs this node from its first line —
    # not from the interrupt() call. interrupt() then returns the resume value
    # instead of pausing. Everything above is a pure read of state, so executing
    # it a second time changes nothing.
    #
    # Never put a side effect above this line. An audit insert, a Slack ping, a
    # counter bump placed there would fire once when the graph pauses and a
    # second time when it resumes. That is why approval lives in its own node
    # and the side effects live in `execute` — which contains no interrupt and
    # therefore never replays mid-flight.
    # ------------------------------------------------------------------------
    answer = interrupt(payload)

    decision = _normalize(answer)
    review: Review = {
        "action": decision.action.value,
        "actor": decision.actor,
        "reason": decision.reason,
        "decided_at": _utcnow_iso(),
        "args_modified": False,
    }

    updated = dict(pending)
    if decision.action is ResumeAction.EDIT:
        final_args = dict(decision.args or {})
        updated["final_args"] = final_args
        review["args_modified"] = final_args != pending["proposed_args"]

    return {"pending": updated, "review": review}


def _rationale(state: AgentState) -> str:
    """The model's own words about what it is proposing — shown to the reviewer."""
    latest = _last_ai_message(state["messages"])
    if latest is None:
        return ""
    content = latest.content
    if isinstance(content, str):
        return content.strip()
    parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(part for part in parts if part).strip()


def _normalize(answer: Any) -> ResumePayload:
    """Accept a resume value from any caller and coerce it to the 3-way shape.

    The API validates before resuming, but the graph can also be driven
    directly from Python or the LangGraph CLI, so it re-validates here. An
    unparseable answer is treated as a rejection: when a human's intent is
    unclear, the safe reading is 'do not act'.
    """
    if isinstance(answer, ResumePayload):
        return answer
    if isinstance(answer, str):
        answer = {"action": answer}
    if not isinstance(answer, dict):
        return ResumePayload(
            action=ResumeAction.REJECT,
            reason=f"Unreadable resume payload of type {type(answer).__name__}.",
            actor="system",
        )
    try:
        return ResumePayload.model_validate(answer)
    except Exception as exc:
        return ResumePayload(
            action=ResumeAction.REJECT,
            reason=f"Invalid resume payload: {exc}",
            actor="system",
        )


async def execute(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Run the tool. The only node in the graph that causes side effects."""
    pending = state["pending"]
    assert pending is not None, "execute reached with no pending call"
    review = state.get("review")

    thread_id = _thread_id(config)
    args = pending["final_args"]
    tool_call_id = pending["tool_call_id"]

    if review is None:
        decision, decided_by, decision_reason = "auto_approved", "policy:auto", pending["policy_reason"]
        decided_at = None
    else:
        decision = "edited" if review["args_modified"] else "approved"
        decided_by = review["actor"]
        decision_reason = review["reason"]
        decided_at = _parse_iso(review["decided_at"])

    # If the process died between the tool running and the checkpoint being
    # written, LangGraph replays this node on restart. The audit table is the
    # durable record of what already happened, so we ask it first.
    if await audit.already_executed(tool_call_id):
        content = (
            "This call already completed in an earlier attempt and was not run "
            "again. Treat the original result as final."
        )
        await audit.record(
            thread_id=thread_id,
            tool_call_id=tool_call_id,
            tool_name=pending["name"],
            policy_mode=pending["mode"],
            policy_source=pending["policy_source"],
            policy_reason=pending["policy_reason"],
            decision=decision,
            decided_by=decided_by,
            decision_reason=decision_reason,
            proposed_args=pending["proposed_args"],
            final_args=args,
            status="skipped_duplicate",
            result_summary=content,
            requested_at=_parse_iso(pending["requested_at"]),
            decided_at=decided_at,
            completed_at=audit.utcnow(),
        )
        return _resolve(tool_call_id, content)

    started = time.perf_counter()
    error = ""
    try:
        content = await run_tool(pending["name"], args)
        status = "executed"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        content = f"ERROR: the tool failed. {error}"
        status = "failed"
    duration_ms = int((time.perf_counter() - started) * 1000)

    await audit.record(
        thread_id=thread_id,
        tool_call_id=tool_call_id,
        tool_name=pending["name"],
        policy_mode=pending["mode"],
        policy_source=pending["policy_source"],
        policy_reason=pending["policy_reason"],
        decision=decision,
        decided_by=decided_by,
        decision_reason=decision_reason,
        proposed_args=pending["proposed_args"],
        final_args=args,
        status=status,
        result_summary=content,
        error=error,
        requested_at=_parse_iso(pending["requested_at"]),
        decided_at=decided_at,
        completed_at=audit.utcnow(),
        duration_ms=duration_ms,
    )
    return _resolve(tool_call_id, content)


async def rejected(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """A human said no. Tell the model, audit it, carry on."""
    pending = state["pending"]
    review = state.get("review") or {}
    assert pending is not None, "rejected reached with no pending call"

    reason = review.get("reason") or "No reason given."
    actor = review.get("actor", "unknown")
    content = (
        f"BLOCKED: a human reviewer rejected this call to `{pending['name']}`. "
        f"Reason: {reason} Do not retry it. Tell the user it was not approved "
        f"and suggest an alternative if there is one."
    )

    await audit.record(
        thread_id=_thread_id(config),
        tool_call_id=pending["tool_call_id"],
        tool_name=pending["name"],
        policy_mode=pending["mode"],
        policy_source=pending["policy_source"],
        policy_reason=pending["policy_reason"],
        decision="rejected",
        decided_by=actor,
        decision_reason=reason,
        proposed_args=pending["proposed_args"],
        final_args=pending["proposed_args"],
        status="blocked",
        result_summary=content,
        requested_at=_parse_iso(pending["requested_at"]),
        decided_at=_parse_iso(review.get("decided_at")),
        completed_at=audit.utcnow(),
    )
    return _resolve(pending["tool_call_id"], content)


async def denied(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Policy said no. No human is ever bothered."""
    pending = state["pending"]
    assert pending is not None, "denied reached with no pending call"

    content = (
        f"BLOCKED BY POLICY: `{pending['name']}` is not permitted. "
        f"{pending['policy_reason']} Do not retry it. Tell the user this "
        f"capability is unavailable."
    )

    await audit.record(
        thread_id=_thread_id(config),
        tool_call_id=pending["tool_call_id"],
        tool_name=pending["name"],
        policy_mode=pending["mode"],
        policy_source=pending["policy_source"],
        policy_reason=pending["policy_reason"],
        decision="denied",
        decided_by="policy",
        decision_reason=pending["policy_reason"],
        proposed_args=pending["proposed_args"],
        final_args=pending["proposed_args"],
        status="blocked",
        result_summary=content,
        requested_at=_parse_iso(pending["requested_at"]),
        completed_at=audit.utcnow(),
    )
    return _resolve(pending["tool_call_id"], content)


def _resolve(tool_call_id: str, content: str) -> dict[str, Any]:
    """Close out a pending call with a ToolMessage and clear the gate state."""
    return {
        "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
        "pending": None,
        "review": None,
    }


def _parse_iso(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value)
    except ValueError:
        return None


# --- routing -----------------------------------------------------------------


def route_from_agent(state: AgentState) -> str:
    """Did the model ask for a tool, or is it done?"""
    return "policy_gate" if next_pending_call(state["messages"]) else "__end__"


def route_from_gate(state: AgentState) -> str:
    """The fork the whole project is about."""
    pending = state.get("pending")
    if pending is None:
        # Every tool call from this turn is resolved; let the model see them.
        return "agent"

    mode = Mode(pending["mode"])
    if mode is Mode.AUTO:
        return "execute"
    if mode is Mode.APPROVE:
        return "human_review"
    return "denied"


def route_from_review(state: AgentState) -> str:
    review = state.get("review")
    if review is None or review["action"] == ResumeAction.REJECT.value:
        return "rejected"
    return "execute"
