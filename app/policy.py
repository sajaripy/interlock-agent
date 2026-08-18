"""The policy layer.

Every tool call is evaluated here before dispatch. This is what makes the
project a system rather than a demo with a hardcoded `if tool == "send_email"`:
the risk tiers live in a YAML file, validated by Pydantic, and the graph
consults them at runtime.

Two rules matter:

1. Default-deny. A tool that is not named in the policy file is blocked.
2. Floors win. Each tool declares, in code, the least restrictive mode it
   will ever accept. A policy file cannot downgrade `send_email` to `auto`.
   Configuration is allowed to make the system stricter, never looser.
"""

from __future__ import annotations

import threading
from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings


class Mode(str, Enum):
    """The three risk tiers, ordered from least to most restrictive."""

    AUTO = "auto"
    APPROVE = "approve"
    DENY = "deny"


# Severity ranking, used to apply floors.
_RANK: dict[Mode, int] = {Mode.AUTO: 0, Mode.APPROVE: 1, Mode.DENY: 2}


def stricter(a: Mode, b: Mode) -> Mode:
    return a if _RANK[a] >= _RANK[b] else b


class ToolRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Mode
    reason: str = ""


class PolicyConfig(BaseModel):
    """The parsed policy.yaml."""

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    hot_reload: bool = True
    default: Mode = Mode.DENY
    tools: dict[str, ToolRule] = Field(default_factory=dict)


class PolicyDecision(BaseModel):
    """The result of evaluating one tool against the policy."""

    tool: str
    mode: Mode
    reason: str
    source: str
    floor_applied: bool = False

    @property
    def requires_human(self) -> bool:
        return self.mode is Mode.APPROVE

    @property
    def blocked(self) -> bool:
        return self.mode is Mode.DENY


def load_policy(path: Path | None = None) -> PolicyConfig:
    """Read and validate policy.yaml. Raises on a malformed file."""
    path = path or get_settings().policy_file
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return PolicyConfig.model_validate(raw)


class PolicyEngine:
    """Holds the policy and evaluates tool calls against it.

    When `hot_reload` is on, the file's mtime is checked on each evaluation so
    an operator can tighten a rule while a run is parked at an approval.
    """

    def __init__(self, path: Path | None = None, config: PolicyConfig | None = None):
        self._path = path or get_settings().policy_file
        self._lock = threading.Lock()
        self._mtime: float | None = None
        if config is not None:
            self._config = config
            self._mtime = None
        else:
            self._config = load_policy(self._path)
            self._mtime = self._path.stat().st_mtime

    @property
    def config(self) -> PolicyConfig:
        return self._config

    def _maybe_reload(self) -> None:
        if not self._config.hot_reload or self._mtime is None:
            return
        try:
            mtime = self._path.stat().st_mtime
        except OSError:
            return
        if mtime == self._mtime:
            return
        with self._lock:
            try:
                self._config = load_policy(self._path)
                self._mtime = mtime
            except Exception:
                # A broken edit must not take the gate offline. Keep serving
                # the last good policy; the next valid save picks up.
                self._mtime = mtime

    def evaluate(self, tool_name: str, floor: Mode = Mode.AUTO) -> PolicyDecision:
        """Decide how `tool_name` should be dispatched.

        `floor` is the tool's own declared minimum, from the code registry.
        """
        self._maybe_reload()

        rule = self._config.tools.get(tool_name)
        if rule is None:
            configured = self._config.default
            reason = f"No rule for '{tool_name}'; falling back to default ({configured.value})."
            source = "policy:default"
        else:
            configured = rule.mode
            reason = rule.reason or f"policy.yaml: tools.{tool_name}.mode = {configured.value}"
            source = f"policy:tools.{tool_name}"

        effective = stricter(configured, floor)
        floor_applied = effective is not configured
        if floor_applied:
            reason = (
                f"{reason} Escalated from '{configured.value}' to '{effective.value}' "
                f"by the code-declared floor for this tool."
            )
            source = f"{source}+floor"

        return PolicyDecision(
            tool=tool_name,
            mode=effective,
            reason=reason,
            source=source,
            floor_applied=floor_applied,
        )


_engine: PolicyEngine | None = None


def get_policy_engine() -> PolicyEngine:
    global _engine
    if _engine is None:
        _engine = PolicyEngine()
    return _engine


def set_policy_engine(engine: PolicyEngine | None) -> None:
    """Swap the process-wide engine. Used by tests and by app startup."""
    global _engine
    _engine = engine
