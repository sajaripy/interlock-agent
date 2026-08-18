"""The tool registry: one entry per tool, each carrying its risk floor.

The floor is the least restrictive mode the policy file is allowed to grant.
It lives in code, next to the tool, because it is a property of what the tool
*does* — not an operational preference. `send_email` reaches a human being; no
YAML edit should be able to make that automatic.

The policy file can still make any tool stricter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool

from app.policy import Mode
from app.tools.db_write import create_ticket
from app.tools.email import send_email
from app.tools.http_get import fetch_url


@dataclass(frozen=True)
class ToolSpec:
    tool: BaseTool
    floor: Mode
    effect: str
    reversible: bool

    @property
    def name(self) -> str:
        return self.tool.name


TOOLS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec(
            tool=fetch_url,
            floor=Mode.AUTO,
            effect="Reads a public URL. Nothing changes.",
            reversible=True,
        ),
        ToolSpec(
            tool=create_ticket,
            floor=Mode.APPROVE,
            effect="Inserts a row into the tickets table.",
            reversible=True,
        ),
        ToolSpec(
            tool=send_email,
            floor=Mode.APPROVE,
            effect="Delivers a message to an external recipient.",
            reversible=False,
        ),
    )
}


def all_tools() -> list[BaseTool]:
    """The tool objects to bind to the model."""
    return [spec.tool for spec in TOOLS.values()]


def get_spec(name: str) -> ToolSpec | None:
    return TOOLS.get(name)


def floor_for(name: str) -> Mode:
    """An unknown tool floors at DENY — the model cannot invent capabilities."""
    spec = TOOLS.get(name)
    return spec.floor if spec else Mode.DENY


async def run_tool(name: str, args: dict[str, Any]) -> str:
    """Dispatch a tool by name. Only ever called after the policy gate."""
    spec = TOOLS.get(name)
    if spec is None:
        raise KeyError(f"Unknown tool '{name}'.")
    result = await spec.tool.ainvoke(args)
    return result if isinstance(result, str) else str(result)
