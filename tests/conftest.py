"""Test fixtures.

The suite runs with no Postgres, no SMTP server and no API key:
  * the checkpointer is MemorySaver (the only place it is allowed);
  * the application tables live in a throwaway SQLite file;
  * the model is a scripted stand-in that replays canned tool calls;
  * `send_email`'s delivery function is replaced by a recorder, so a test can
    assert the difference between "the agent wanted to send" and "an email
    actually went out".
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

# Must happen before anything imports app.config.
_TMP = Path(tempfile.mkdtemp(prefix="interlock-tests-"))
os.environ["APP_DB_URL"] = f"sqlite+aiosqlite:///{(_TMP / 'test.db').as_posix()}"
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")

import pytest  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage  # noqa: E402

from app import db, llm  # noqa: E402
from app.graph.build import memory_graph  # noqa: E402
from app.llm import ScriptedChatModel, tool_call_message  # noqa: E402
from app.policy import set_policy_engine  # noqa: E402

# The tests build tool calls constantly; keep the short local alias.
ai_tool_call = tool_call_message


class Recorder:
    """Stands in for an irreversible side effect and remembers if it happened."""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def called(self) -> bool:
        return bool(self.calls)

    async def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(args[0] if args else kwargs)


@pytest.fixture(autouse=True)
async def fresh_database():
    """A clean set of application tables for every test."""
    await db.dispose_db()
    engine = db.get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(db.Base.metadata.drop_all)
        await conn.run_sync(db.Base.metadata.create_all)
    yield
    await db.dispose_db()


@pytest.fixture(autouse=True)
def reset_globals():
    yield
    llm.set_llm(None)
    set_policy_engine(None)


@pytest.fixture
def script():
    """Install a scripted model and hand the test its script list."""

    def _install(*responses: BaseMessage) -> ScriptedChatModel:
        model = ScriptedChatModel(responses=list(responses), calls=[])
        llm.set_llm(model)
        return model

    return _install


@pytest.fixture
def no_delivery(monkeypatch) -> Recorder:
    """Replace SMTP delivery with a recorder."""
    recorder = Recorder()
    monkeypatch.setattr("app.tools.email._deliver", recorder)
    return recorder


@pytest.fixture
def graph():
    return memory_graph()


@pytest.fixture
def thread():
    counter = {"n": 0}

    def _next() -> dict[str, Any]:
        counter["n"] += 1
        return {"configurable": {"thread_id": f"test-thread-{counter['n']}"}}

    return _next
