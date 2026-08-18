"""Graph state and the shape of a resume payload.

Everything in `AgentState` is checkpointed to Postgres after every node, so it
must be JSON-serializable. That is why `pending` is a TypedDict of plain values
rather than a Pydantic model or a dataclass.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, model_validator


class PendingCall(TypedDict):
    """A tool call that has been evaluated by policy but not yet resolved."""

    tool_call_id: str
    name: str
    proposed_args: dict[str, Any]
    final_args: dict[str, Any]
    mode: str
    policy_source: str
    policy_reason: str
    floor_applied: bool
    effect: str
    reversible: bool
    requested_at: str


class Review(TypedDict):
    """A human's answer, normalized."""

    action: str
    actor: str
    reason: str
    decided_at: str
    args_modified: bool


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    pending: NotRequired[PendingCall | None]
    review: NotRequired[Review | None]


# --- the resume payload ------------------------------------------------------


class ResumeAction(str, Enum):
    """Three-way, not two-way.

    Binary approve/reject is where most examples stop. In practice a reviewer
    usually agrees with *what* the agent wants to do and disagrees with the
    details — wrong recipient, too-strong priority, a subject line that reads
    badly. EDIT lets them fix it in place instead of rejecting and re-prompting.
    """

    APPROVE = "approve"
    REJECT = "reject"
    EDIT = "edit"


class ResumePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: ResumeAction
    args: dict[str, Any] | None = None
    reason: str = ""
    actor: str = "unknown"

    @model_validator(mode="after")
    def _check_args(self) -> "ResumePayload":
        if self.action is ResumeAction.EDIT and self.args is None:
            raise ValueError("action 'edit' requires an 'args' object.")
        if self.action is not ResumeAction.EDIT and self.args is not None:
            raise ValueError("'args' is only valid with action 'edit'.")
        return self
