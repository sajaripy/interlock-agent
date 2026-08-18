"""The policy layer, including the rule that configuration can only tighten."""

from __future__ import annotations

import textwrap

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from app.policy import (
    Mode,
    PolicyConfig,
    PolicyEngine,
    ToolRule,
    load_policy,
    set_policy_engine,
)
from app.tools import TOOLS, floor_for
from tests.conftest import ai_tool_call


def engine_with(**tools: Mode) -> PolicyEngine:
    return PolicyEngine(
        config=PolicyConfig(
            default=Mode.DENY,
            hot_reload=False,
            tools={name: ToolRule(mode=mode) for name, mode in tools.items()},
        )
    )


# --- the shipped policy file -------------------------------------------------


def test_shipped_policy_file_parses():
    config = load_policy()
    assert config.default is Mode.DENY
    assert set(config.tools) == {"fetch_url", "create_ticket", "send_email"}


def test_every_registered_tool_has_a_rule():
    """A tool with no rule is dead on arrival, so catch it here rather than
    at 2am when the agent reports a capability it cannot use."""
    config = load_policy()
    assert set(TOOLS) <= set(config.tools)


def test_malformed_policy_is_rejected(tmp_path):
    bad = tmp_path / "policy.yaml"
    bad.write_text(
        textwrap.dedent(
            """
            version: 1
            default: deny
            tools:
              send_email:
                mode: sometimes
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_policy(bad)


def test_unknown_keys_are_rejected(tmp_path):
    bad = tmp_path / "policy.yaml"
    bad.write_text("version: 1\ndefualt: deny\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_policy(bad)


# --- evaluation --------------------------------------------------------------


def test_unlisted_tool_falls_back_to_default_deny():
    decision = engine_with().evaluate("some_new_tool", floor=Mode.AUTO)
    assert decision.mode is Mode.DENY
    assert decision.blocked
    assert decision.source == "policy:default"


def test_read_only_tool_is_auto():
    decision = engine_with(fetch_url=Mode.AUTO).evaluate("fetch_url", floor=floor_for("fetch_url"))
    assert decision.mode is Mode.AUTO
    assert not decision.requires_human


def test_db_write_requires_approval():
    decision = engine_with(create_ticket=Mode.APPROVE).evaluate(
        "create_ticket", floor=floor_for("create_ticket")
    )
    assert decision.requires_human


# --- floors ------------------------------------------------------------------


def test_policy_cannot_downgrade_email_to_auto():
    """The headline safety property: a YAML typo cannot disarm the interlock."""
    decision = engine_with(send_email=Mode.AUTO).evaluate(
        "send_email", floor=floor_for("send_email")
    )
    assert decision.mode is Mode.APPROVE
    assert decision.floor_applied
    assert "floor" in decision.source


def test_policy_can_still_tighten_past_the_floor():
    """Floors set a minimum, not a maximum. Deny always wins."""
    decision = engine_with(send_email=Mode.DENY).evaluate(
        "send_email", floor=floor_for("send_email")
    )
    assert decision.mode is Mode.DENY
    assert not decision.floor_applied


def test_unknown_tools_floor_at_deny():
    assert floor_for("rm_minus_rf") is Mode.DENY


async def test_floor_escalation_is_enforced_end_to_end(graph, script, no_delivery, thread):
    """Not just the unit: a misconfigured file still pauses the real graph."""
    set_policy_engine(engine_with(send_email=Mode.AUTO))

    script(
        ai_tool_call("send_email", {"to": "x@example.com", "subject": "S", "body": "B"}, "call-1"),
        AIMessage(content="Sent."),
    )
    config = thread()

    result = await graph.ainvoke({"messages": [HumanMessage(content="Email x.")]}, config)

    assert "__interrupt__" in result, "policy said auto, but the floor should have paused it"
    assert not no_delivery.called
    assert result["__interrupt__"][0].value["floor_applied"] is True


# --- hot reload --------------------------------------------------------------


def test_policy_reloads_when_the_file_changes(tmp_path):
    path = tmp_path / "policy.yaml"
    path.write_text(
        "version: 1\nhot_reload: true\ndefault: deny\ntools:\n  fetch_url:\n    mode: auto\n",
        encoding="utf-8",
    )
    engine = PolicyEngine(path=path)
    assert engine.evaluate("fetch_url").mode is Mode.AUTO

    # An operator tightens the rule while the process is running.
    import os
    import time

    path.write_text(
        "version: 1\nhot_reload: true\ndefault: deny\ntools:\n  fetch_url:\n    mode: deny\n",
        encoding="utf-8",
    )
    os.utime(path, (time.time() + 1, time.time() + 1))

    assert engine.evaluate("fetch_url").mode is Mode.DENY


def test_a_broken_edit_keeps_serving_the_last_good_policy(tmp_path):
    """A syntax error in the policy file must not take the gate offline."""
    path = tmp_path / "policy.yaml"
    path.write_text(
        "version: 1\nhot_reload: true\ndefault: deny\ntools:\n  fetch_url:\n    mode: auto\n",
        encoding="utf-8",
    )
    engine = PolicyEngine(path=path)
    assert engine.evaluate("fetch_url").mode is Mode.AUTO

    import os
    import time

    path.write_text("this: is: not: valid: yaml:\n", encoding="utf-8")
    os.utime(path, (time.time() + 1, time.time() + 1))

    assert engine.evaluate("fetch_url").mode is Mode.AUTO
