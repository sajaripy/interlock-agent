"""The approval queue.

A thin client over the HTTP API — it holds no state of its own and knows
nothing about LangGraph. The run lives in Postgres; this page just renders
what `GET /runs/{thread_id}` reports and posts a decision back. You could
close it mid-approval, reopen it tomorrow, and the pending call would still
be there.

Run with:  streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import streamlit as st

DEFAULT_API = "http://localhost:8000"

st.set_page_config(page_title="Interlock", page_icon="🔐", layout="wide")


# --- API client --------------------------------------------------------------


def api(path: str) -> str:
    return f"{st.session_state.api_base.rstrip('/')}{path}"


def call(method: str, path: str, **kwargs: Any) -> tuple[bool, Any]:
    try:
        response = httpx.request(method, api(path), timeout=180.0, **kwargs)
    except httpx.HTTPError as exc:
        return False, f"Could not reach the API at {st.session_state.api_base}: {exc}"
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        return False, f"HTTP {response.status_code}: {detail}"
    return True, response.json()


# --- state -------------------------------------------------------------------

st.session_state.setdefault("api_base", DEFAULT_API)
st.session_state.setdefault("thread_id", None)
st.session_state.setdefault("run", None)
st.session_state.setdefault("error", None)


def refresh() -> None:
    if not st.session_state.thread_id:
        return
    ok, data = call("GET", f"/runs/{st.session_state.thread_id}")
    if ok:
        st.session_state.run = data
    else:
        st.session_state.error = data


# --- sidebar -----------------------------------------------------------------

with st.sidebar:
    st.title("🔐 Interlock")
    st.caption("An agent that stops before it does anything risky.")

    st.session_state.api_base = st.text_input("API base URL", st.session_state.api_base)
    actor = st.text_input("Reviewing as", value="you@example.com")

    st.divider()
    st.subheader("Current policy")
    ok, policy = call("GET", "/policy")
    if ok:
        badge = {"auto": "🟢", "approve": "🟡", "deny": "🔴"}
        for rule in policy["rules"]:
            mark = badge.get(rule["effective_mode"], "⚪")
            st.markdown(f"{mark} **{rule['tool']}** — `{rule['effective_mode']}`")
            st.caption(rule["effect"])
            if rule["configured_mode"] != rule["effective_mode"]:
                st.warning(
                    f"policy.yaml says `{rule['configured_mode']}`, raised to "
                    f"`{rule['effective_mode']}` by this tool's floor.",
                    icon="⚠️",
                )
    else:
        st.error(policy)

    st.divider()
    if st.session_state.thread_id:
        st.caption("Thread")
        st.code(st.session_state.thread_id, language=None)
        if st.button("Refresh", use_container_width=True):
            refresh()
        if st.button("New run", use_container_width=True):
            st.session_state.thread_id = None
            st.session_state.run = None
            st.rerun()

    st.divider()
    st.caption("Resume an existing run")
    with st.form("resume_existing", clear_on_submit=True):
        existing = st.text_input("thread_id", label_visibility="collapsed", placeholder="run-...")
        if st.form_submit_button("Load") and existing:
            st.session_state.thread_id = existing.strip()
            refresh()
            st.rerun()


# --- main --------------------------------------------------------------------

st.header("Approval queue")

if st.session_state.error:
    st.error(st.session_state.error)
    st.session_state.error = None

# Start a run.
if not st.session_state.thread_id:
    st.write("Ask the agent to do something. Anything risky will come back here for review.")
    with st.form("start"):
        message = st.text_area(
            "Request",
            placeholder="Email oncall@example.com about the disk alert on node 3.",
            height=100,
        )
        if st.form_submit_button("Run", type="primary") and message.strip():
            with st.spinner("The agent is working..."):
                ok, data = call("POST", "/runs", json={"message": message})
            if ok:
                st.session_state.thread_id = data["thread_id"]
                st.session_state.run = data
                st.rerun()
            else:
                st.session_state.error = data
                st.rerun()

    st.info(
        "Try: *“Email oncall@example.com about the disk alert”* (needs approval), "
        "or *“What does https://example.com say?”* (runs on its own).",
        icon="💡",
    )
    st.stop()


run = st.session_state.run or {}
conversation, trail = st.tabs(["Conversation", "Audit trail"])


with conversation:
    # --- the pending approval, if any ---------------------------------------
    approval = run.get("approval")
    if approval:
        st.subheader("⏸️ Waiting for your decision")

        left, right = st.columns([2, 1])
        with right:
            st.metric("Tool", approval["tool"])
            st.metric("Reversible", "no" if not approval["reversible"] else "yes")
            st.caption(approval["effect"])
            with st.expander("Why this needs approval"):
                st.write(approval["policy_reason"])
                st.caption(f"Source: `{approval['policy_source']}`")

        with left:
            if approval.get("agent_rationale"):
                st.markdown("**The agent says:**")
                st.info(approval["agent_rationale"])

            st.markdown("**Proposed arguments** — edit them before approving if you like:")
            edited = st.text_area(
                "args",
                value=json.dumps(approval["args"], indent=2),
                height=220,
                label_visibility="collapsed",
                key=f"args-{approval['tool_call_id']}",
            )
            reason = st.text_input(
                "Note for the audit log",
                placeholder="Why you approved, edited, or rejected this.",
            )

            parsed: dict[str, Any] | None = None
            changed = False
            try:
                parsed = json.loads(edited)
                changed = parsed != approval["args"]
                if changed:
                    st.caption("✏️ Arguments differ from what the agent proposed.")
            except json.JSONDecodeError as exc:
                st.error(f"That isn't valid JSON: {exc}")

            approve_col, edit_col, reject_col = st.columns(3)

            def decide(payload: dict[str, Any]) -> None:
                with st.spinner("Resuming the run..."):
                    ok, data = call(
                        "POST",
                        f"/runs/{st.session_state.thread_id}/resume",
                        json=payload,
                    )
                if ok:
                    st.session_state.run = data
                else:
                    st.session_state.error = data
                st.rerun()

            with approve_col:
                if st.button(
                    "✅ Approve as proposed",
                    use_container_width=True,
                    type="primary",
                    disabled=changed,
                    help="Disabled while the arguments differ — use Approve edited instead.",
                ):
                    decide({"action": "approve", "actor": actor, "reason": reason})

            with edit_col:
                if st.button(
                    "✏️ Approve edited",
                    use_container_width=True,
                    disabled=not changed or parsed is None,
                    help="Runs the tool with your edited arguments.",
                ):
                    decide({"action": "edit", "args": parsed, "actor": actor, "reason": reason})

            with reject_col:
                if st.button("🚫 Reject", use_container_width=True):
                    decide({"action": "reject", "actor": actor, "reason": reason})

        st.divider()

    elif run.get("status") == "completed":
        st.success("Run complete — nothing is waiting.")
        if run.get("reply"):
            st.markdown("**Agent's answer:**")
            st.write(run["reply"])
        st.divider()

    # --- transcript ----------------------------------------------------------
    st.subheader("Transcript")
    icons = {"human": "🧑", "ai": "🤖", "tool": "🔧", "system": "⚙️"}
    for message in run.get("messages", []):
        with st.chat_message(message["role"], avatar=icons.get(message["role"])):
            if message["content"]:
                st.write(message["content"])
            for call_ in message.get("tool_calls") or []:
                st.caption(f"proposed `{call_['name']}`")
                st.json(call_["args"], expanded=False)

    # --- continue the conversation -------------------------------------------
    if not approval:
        follow_up = st.chat_input("Ask a follow-up on this thread")
        if follow_up:
            with st.spinner("The agent is working..."):
                ok, data = call(
                    "POST",
                    "/runs",
                    json={"message": follow_up, "thread_id": st.session_state.thread_id},
                )
            if ok:
                st.session_state.run = data
            else:
                st.session_state.error = data
            st.rerun()


with trail:
    st.subheader("Every gated decision on this run")
    st.caption(
        "One row per tool call, whether it ran, was rejected, or was blocked. "
        "`proposed` is what the agent wanted; `final` is what actually ran."
    )
    ok, entries = call("GET", f"/runs/{st.session_state.thread_id}/audit")
    if not ok:
        st.error(entries)
    elif not entries:
        st.info("Nothing gated yet on this thread.")
    else:
        marks = {
            "auto_approved": "🟢 auto",
            "approved": "✅ approved",
            "edited": "✏️ edited",
            "rejected": "🚫 rejected",
            "denied": "🔴 denied by policy",
        }
        for entry in reversed(entries):
            label = (
                f"{marks.get(entry['decision'], entry['decision'])} · "
                f"`{entry['tool_name']}` · {entry['status']}"
            )
            with st.expander(label, expanded=entry["args_modified"]):
                meta = st.columns(3)
                meta[0].caption(f"**Decided by**\n\n{entry['decided_by']}")
                meta[1].caption(f"**Policy mode**\n\n`{entry['policy_mode']}`")
                meta[2].caption(f"**When**\n\n{entry['completed_at'] or entry['requested_at']}")

                if entry["decision_reason"]:
                    st.caption(f"Note: {entry['decision_reason']}")

                if entry["args_modified"]:
                    before, after = st.columns(2)
                    before.markdown("**Agent proposed**")
                    before.json(entry["proposed_args"])
                    after.markdown("**Human approved**")
                    after.json(entry["final_args"])
                else:
                    st.markdown("**Arguments**")
                    st.json(entry["final_args"])

                if entry["error"]:
                    st.error(entry["error"])
                elif entry["result_summary"]:
                    st.code(entry["result_summary"][:2000], language=None)
