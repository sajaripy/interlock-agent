"""Writes and reads the audit trail.

Every gated tool call produces exactly one `audit_log` row, whether it ran,
was rejected by a human, or was blocked by policy. The row is written by the
graph node that resolved the call, so there is no path through the graph that
performs an action without leaving a record.

`already_executed` gives the executor an idempotency check: if the process dies
between running a tool and writing its checkpoint, LangGraph will re-run that
node on restart. The audit table is the durable record of "this side effect
already happened", so a crash mid-send cannot send the email twice.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Sequence

from sqlalchemy import select

from app.db import AuditLog, get_sessionmaker


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


async def record(
    *,
    thread_id: str,
    tool_call_id: str,
    tool_name: str,
    policy_mode: str,
    policy_source: str = "",
    policy_reason: str = "",
    decision: str,
    decided_by: str = "system",
    decision_reason: str = "",
    proposed_args: dict[str, Any],
    final_args: dict[str, Any],
    status: str,
    result_summary: str = "",
    error: str = "",
    requested_at: dt.datetime | None = None,
    decided_at: dt.datetime | None = None,
    completed_at: dt.datetime | None = None,
    duration_ms: int | None = None,
) -> int:
    """Insert one audit row. Returns its id."""
    entry = AuditLog(
        thread_id=thread_id,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        policy_mode=policy_mode,
        policy_source=policy_source,
        policy_reason=policy_reason,
        decision=decision,
        decided_by=decided_by,
        decision_reason=decision_reason,
        proposed_args=proposed_args,
        final_args=final_args,
        args_modified=proposed_args != final_args,
        status=status,
        result_summary=result_summary[:4000],
        error=error[:4000],
        requested_at=requested_at or utcnow(),
        decided_at=decided_at,
        completed_at=completed_at,
        duration_ms=duration_ms,
    )
    async with get_sessionmaker()() as session:
        session.add(entry)
        await session.commit()
        return entry.id


async def already_executed(tool_call_id: str) -> bool:
    """True if this exact tool call has already run to completion.

    Guards against LangGraph replaying an executor node after a crash.
    """
    async with get_sessionmaker()() as session:
        stmt = (
            select(AuditLog.id)
            .where(AuditLog.tool_call_id == tool_call_id)
            .where(AuditLog.status == "executed")
            .limit(1)
        )
        return (await session.execute(stmt)).first() is not None


async def for_thread(thread_id: str) -> Sequence[AuditLog]:
    async with get_sessionmaker()() as session:
        stmt = (
            select(AuditLog)
            .where(AuditLog.thread_id == thread_id)
            .order_by(AuditLog.id.asc())
        )
        return (await session.execute(stmt)).scalars().all()


async def recent(limit: int = 100) -> Sequence[AuditLog]:
    async with get_sessionmaker()() as session:
        stmt = select(AuditLog).order_by(AuditLog.id.desc()).limit(limit)
        return (await session.execute(stmt)).scalars().all()


def to_dict(entry: AuditLog) -> dict[str, Any]:
    return {
        "id": entry.id,
        "thread_id": entry.thread_id,
        "tool_call_id": entry.tool_call_id,
        "tool_name": entry.tool_name,
        "policy_mode": entry.policy_mode,
        "policy_source": entry.policy_source,
        "policy_reason": entry.policy_reason,
        "decision": entry.decision,
        "decided_by": entry.decided_by,
        "decision_reason": entry.decision_reason,
        "proposed_args": entry.proposed_args,
        "final_args": entry.final_args,
        "args_modified": bool(entry.args_modified),
        "status": entry.status,
        "result_summary": entry.result_summary,
        "error": entry.error,
        "requested_at": entry.requested_at.isoformat() if entry.requested_at else None,
        "decided_at": entry.decided_at.isoformat() if entry.decided_at else None,
        "completed_at": entry.completed_at.isoformat() if entry.completed_at else None,
        "duration_ms": entry.duration_ms,
    }
