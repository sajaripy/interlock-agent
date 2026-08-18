"""SQLAlchemy 2.0 async database layer for the application tables.

This is separate from the LangGraph checkpointer, which owns its own tables in
the same Postgres database and speaks psycopg rather than SQLAlchemy. Keeping
them apart means the agent's memory and the business data never contend for the
same session.

Two tables:
  * `tickets`   — what the DB-write tool actually writes. Stands in for
                  whatever real business object your agent would touch.
  * `audit_log` — the record of every gated decision. See app/audit.py.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import get_settings

# JSONB on Postgres, plain JSON everywhere else (so the tests can use SQLite).
JSONType = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


class Ticket(Base):
    """The thing the `create_ticket` tool writes. A stand-in for real data."""

    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False, default="")
    priority: Mapped[str] = mapped_column(String(20), nullable=False, default="normal")
    created_by: Mapped[str] = mapped_column(String(120), nullable=False, default="agent")
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class AuditLog(Base):
    """One row per gated tool call. Half the value of the whole project.

    It answers, for any action the agent took: what did it want to do, what was
    it allowed to do, who said yes, when, and did they change the arguments
    before it ran.
    """

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # --- what ---------------------------------------------------------------
    thread_id: Mapped[str] = mapped_column(String(120), index=True, nullable=False)
    tool_call_id: Mapped[str] = mapped_column(String(120), index=True, nullable=False)
    tool_name: Mapped[str] = mapped_column(String(120), nullable=False)

    # --- how it was gated ---------------------------------------------------
    policy_mode: Mapped[str] = mapped_column(String(20), nullable=False)
    policy_source: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    policy_reason: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # auto_approved | approved | edited | rejected | denied
    decision: Mapped[str] = mapped_column(String(30), nullable=False)
    decided_by: Mapped[str] = mapped_column(String(120), nullable=False, default="system")
    decision_reason: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # --- the arguments, before and after the human touched them -------------
    proposed_args: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False, default=dict)
    final_args: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False, default=dict)
    args_modified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # --- what happened ------------------------------------------------------
    # executed | failed | blocked | skipped_duplicate
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    result_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    error: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # --- timing -------------------------------------------------------------
    requested_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    decided_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)


# --- engine / session management -------------------------------------------

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        url = get_settings().sqlalchemy_url
        kwargs: dict[str, Any] = {"echo": False, "future": True}
        if not url.startswith("sqlite"):
            kwargs.update(pool_size=5, max_overflow=10, pool_pre_ping=True)
        _engine = create_async_engine(url, **kwargs)
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _sessionmaker


async def init_db() -> None:
    """Create the application tables if they are missing."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose_db() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None
