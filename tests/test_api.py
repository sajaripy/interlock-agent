"""The HTTP surface, driven end to end.

The Postgres lifespan is bypassed here: `ASGITransport` does not run lifespan
events, so the graph dependency is overridden with a MemorySaver-backed graph.
Everything else — routing, validation, serialization — is the real thing.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage

from app.api.main import app, get_graph
from app.graph.build import memory_graph
from tests.conftest import ai_tool_call


@pytest.fixture
def client(graph):
    app.dependency_overrides[get_graph] = lambda: graph
    transport = ASGITransport(app=app)
    yield AsyncClient(transport=transport, base_url="http://test")
    app.dependency_overrides.clear()


EMAIL_CALL = {"to": "ops@example.com", "subject": "Disk full", "body": "Node 3 at 98%."}


async def test_health(client):
    async with client as http:
        response = await http.get("/health")
    assert response.status_code == 200


async def test_start_run_returns_an_approval_request(client, script, no_delivery):
    script(ai_tool_call("send_email", EMAIL_CALL, "call-1", text="Emailing ops."))

    async with client as http:
        response = await http.post("/runs", json={"message": "Email ops about the disk."})

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "awaiting_approval"
    assert body["approval"]["tool"] == "send_email"
    assert body["approval"]["args"] == EMAIL_CALL
    assert body["approval"]["reversible"] is False
    assert body["approval"]["agent_rationale"] == "Emailing ops."
    assert not no_delivery.called


async def test_get_run_shows_what_it_is_waiting_for(client, script, no_delivery):
    script(ai_tool_call("send_email", EMAIL_CALL, "call-1"))

    async with client as http:
        started = (await http.post("/runs", json={"message": "Email ops."})).json()
        thread_id = started["thread_id"]

        fetched = (await http.get(f"/runs/{thread_id}")).json()

    assert fetched["status"] == "awaiting_approval"
    assert fetched["approval"]["tool_call_id"] == "call-1"
    assert fetched["thread_id"] == thread_id


async def test_resume_approve(client, script, no_delivery):
    script(
        ai_tool_call("send_email", EMAIL_CALL, "call-1"),
        AIMessage(content="Sent the email to ops."),
    )

    async with client as http:
        thread_id = (await http.post("/runs", json={"message": "Email ops."})).json()["thread_id"]
        response = await http.post(
            f"/runs/{thread_id}/resume",
            json={"action": "approve", "actor": "alice@example.com"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["reply"] == "Sent the email to ops."
    assert no_delivery.call_count == 1


async def test_resume_edit_changes_the_arguments(client, script, no_delivery):
    script(
        ai_tool_call("send_email", {**EMAIL_CALL, "to": "everyone@example.com"}, "call-1"),
        AIMessage(content="Sent."),
    )

    async with client as http:
        thread_id = (await http.post("/runs", json={"message": "Email everyone."})).json()[
            "thread_id"
        ]
        await http.post(
            f"/runs/{thread_id}/resume",
            json={
                "action": "edit",
                "args": {**EMAIL_CALL, "to": "oncall@example.com"},
                "actor": "alice@example.com",
                "reason": "Narrower recipient.",
            },
        )
        entries = (await http.get(f"/runs/{thread_id}/audit")).json()

    assert no_delivery.calls[0]["To"] == "oncall@example.com"
    assert entries[0]["decision"] == "edited"
    assert entries[0]["args_modified"] is True
    assert entries[0]["proposed_args"]["to"] == "everyone@example.com"
    assert entries[0]["final_args"]["to"] == "oncall@example.com"
    assert entries[0]["decided_by"] == "alice@example.com"


async def test_resume_reject(client, script, no_delivery):
    script(
        ai_tool_call("send_email", EMAIL_CALL, "call-1"),
        AIMessage(content="I did not send it."),
    )

    async with client as http:
        thread_id = (await http.post("/runs", json={"message": "Email ops."})).json()["thread_id"]
        body = (
            await http.post(
                f"/runs/{thread_id}/resume",
                json={"action": "reject", "actor": "bob", "reason": "Not now."},
            )
        ).json()

    assert body["status"] == "completed"
    assert not no_delivery.called


async def test_edit_without_args_is_a_422(client, script, no_delivery):
    script(ai_tool_call("send_email", EMAIL_CALL, "call-1"))

    async with client as http:
        thread_id = (await http.post("/runs", json={"message": "Email ops."})).json()["thread_id"]
        response = await http.post(f"/runs/{thread_id}/resume", json={"action": "edit"})

    assert response.status_code == 422
    assert not no_delivery.called


async def test_resuming_a_run_that_is_not_waiting_is_a_409(client, script):
    script(AIMessage(content="Nothing to do here."))

    async with client as http:
        thread_id = (await http.post("/runs", json={"message": "Hello."})).json()["thread_id"]
        response = await http.post(f"/runs/{thread_id}/resume", json={"action": "approve"})

    assert response.status_code == 409


async def test_unknown_thread_is_a_404(client):
    async with client as http:
        assert (await http.get("/runs/nope")).status_code == 404
        assert (
            await http.post("/runs/nope/resume", json={"action": "approve"})
        ).status_code == 404


async def test_thread_id_is_reusable_for_a_second_turn(client, script):
    """thread_id maps straight onto the checkpoint thread, so conversation
    history carries over."""
    script(
        AIMessage(content="Hello."),
        AIMessage(content="You said hello first."),
    )

    async with client as http:
        first = (await http.post("/runs", json={"message": "Hi."})).json()
        thread_id = first["thread_id"]
        second = (
            await http.post("/runs", json={"message": "What did I say?", "thread_id": thread_id})
        ).json()

    assert second["thread_id"] == thread_id
    roles = [m["role"] for m in second["messages"]]
    assert roles == ["human", "ai", "human", "ai"]


async def test_policy_endpoint_reports_effective_modes(client):
    async with client as http:
        body = (await http.get("/policy")).json()

    rules = {rule["tool"]: rule for rule in body["rules"]}
    assert body["default"] == "deny"
    assert rules["fetch_url"]["effective_mode"] == "auto"
    assert rules["send_email"]["effective_mode"] == "approve"
    assert rules["send_email"]["floor"] == "approve"
    assert rules["send_email"]["reversible"] is False
