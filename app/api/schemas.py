"""Request and response bodies for the HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.graph.state import ResumeAction, ResumePayload


class StartRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(..., min_length=1, description="What the user is asking for.")
    thread_id: str | None = Field(
        default=None,
        description="Reuse a thread to continue a conversation. Generated if omitted.",
    )


class ResumeRequest(ResumePayload):
    """The three-way decision.

    * approve — run it as proposed
    * reject  — do not run it; the agent is told why and continues
    * edit    — run it with `args` instead of what the agent proposed

    Subclasses the graph's own payload model so the API and the graph cannot
    drift apart on what a valid decision looks like; the fields are redeclared
    only to document them in the OpenAPI schema.
    """

    action: ResumeAction
    args: dict[str, Any] | None = Field(
        default=None,
        description="Required for 'edit', forbidden otherwise. Replaces the proposed arguments.",
    )
    reason: str = Field(default="", description="Why. Stored in the audit log.")
    actor: str = Field(default="unknown", description="Who decided. Stored in the audit log.")


class ApprovalRequest(BaseModel):
    """What a reviewer is being asked to decide on."""

    type: str
    tool_call_id: str
    tool: str
    args: dict[str, Any]
    mode: str
    policy_source: str
    policy_reason: str
    floor_applied: bool
    effect: str
    reversible: bool
    requested_at: str
    agent_rationale: str = ""
    actions: list[str] = Field(default_factory=list)


class MessageOut(BaseModel):
    role: Literal["human", "ai", "tool", "system"]
    content: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_call_id: str | None = None


class RunState(BaseModel):
    """The one response shape every run endpoint returns."""

    thread_id: str
    status: Literal["awaiting_approval", "completed"]
    approval: ApprovalRequest | None = None
    reply: str | None = Field(
        default=None, description="The agent's final message, once it has finished."
    )
    messages: list[MessageOut] = Field(default_factory=list)


class AuditEntryOut(BaseModel):
    id: int
    thread_id: str
    tool_call_id: str
    tool_name: str
    policy_mode: str
    policy_source: str
    policy_reason: str
    decision: str
    decided_by: str
    decision_reason: str
    proposed_args: dict[str, Any]
    final_args: dict[str, Any]
    args_modified: bool
    status: str
    result_summary: str
    error: str
    requested_at: str | None
    decided_at: str | None
    completed_at: str | None
    duration_ms: int | None


class PolicyRuleOut(BaseModel):
    tool: str
    configured_mode: str
    floor: str
    effective_mode: str
    reason: str
    effect: str
    reversible: bool


class PolicyOut(BaseModel):
    default: str
    hot_reload: bool
    rules: list[PolicyRuleOut]
