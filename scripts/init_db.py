"""Create both sets of tables: the app's and LangGraph's.

The API does this on startup too, so this script is only for setting things up
ahead of time or checking that your Postgres connection actually works.

    python scripts/init_db.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.db import dispose_db, init_db  # noqa: E402


async def main() -> None:
    settings = get_settings()

    print(f"Application tables -> {settings.sqlalchemy_url}")
    await init_db()
    await dispose_db()
    print("  ok: tickets, audit_log")

    print(f"Checkpoint tables  -> {settings.checkpoint_dsn}")
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async with AsyncPostgresSaver.from_conn_string(settings.checkpoint_dsn) as checkpointer:
        await checkpointer.setup()
    print("  ok: checkpoints, checkpoint_writes, checkpoint_blobs")

    print("\nReady.")


if __name__ == "__main__":
    asyncio.run(main())
