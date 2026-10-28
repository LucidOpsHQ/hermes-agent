"""Real A2A HTTP + native parent/children, replacing only inference/discovery.

A request/reply adapter cannot deliver a detached child after completing its
Task. The inherited push capability used to return a dispatch ACK as help.
"""

import asyncio
import copy
import json
import threading
import urllib.request
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from gateway.config import Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import SessionContext, SessionSource
from hermes_constants import get_hermes_home
from hermes_state import SessionDB
from plugins.platforms.a2a import protocol, tools
from plugins.platforms.a2a.adapter import A2AAdapter
from run_agent import AIAgent


async def until(predicate):
    async with asyncio.timeout(15):
        while not predicate():
            await asyncio.sleep(0.01)


def rpc(base, method, params):
    request = urllib.request.Request(base, json.dumps({
        "jsonrpc": "2.0", "id": "fixture-rpc", "method": method, "params": params,
    }).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def response(text="", call=None):
    message = SimpleNamespace(content=text, tool_calls=[call] if call else None)
    return SimpleNamespace(choices=[SimpleNamespace(
        message=message, finish_reason="tool_calls" if call else "stop")],
        model="test-model", usage=None)


def tool_call(name, args, call_id):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(
        name=name, arguments=json.dumps(args)))


@pytest_asyncio.fixture
async def native_a2a(monkeypatch, tmp_path):
    from tools.async_delegation import has_live_for_session
    from tools.delegate_tool import DELEGATE_TASK_SCHEMA
    from tools.registry import registry
    from tools import approval

    home = get_hermes_home()
    (home / "config.yaml").write_text(
        "approvals:\n  mode: manual\nsecurity:\n  tirith_enabled: false\n"
        "delegation:\n  max_concurrent_children: 3\n  worktree: false\n",
        encoding="utf-8")
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **kw: None)
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", MagicMock())
    monkeypatch.setattr("agent.model_metadata.fetch_model_metadata", lambda *a, **kw: {})
    handler, description, properties, required = tools._TOOLS["a2a_call"]
    schema = {"name": "a2a_call", "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required}}
    registry.register("a2a_call", "a2a", schema, handler)
    sensitive_schema = {"name": "fixture_sensitive_action", "description": "Fixture action",
                        "parameters": {"type": "object", "properties": {}}}
    decisions, options = [], {}
    marker = tmp_path / "must-not-write"

    def sensitive_action(args, **unused):
        decision = approval.request_tool_approval("fixture_sensitive_action", "fixture-only protected action")
        decisions.append(decision)
        if decision["approved"]:
            marker.write_text("approved fixture effect", encoding="utf-8")
        return json.dumps(decision)

    registry.register("fixture_sensitive_action", "a2a", sensitive_schema, sensitive_action)
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *a, **kw: [
        {"type": "function", "function": s} for s in (DELEGATE_TASK_SCHEMA, schema, sensitive_schema)])
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda *a, **kw: {})

    adapter = A2AAdapter(PlatformConfig(enabled=True, extra={"port": 0}))
    # Native capability binding, not a test's hand-picked contextvar value.
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    db = SessionDB(home / "state.db")
    release = threading.Event()
    started, finished, turns, events = set(), set(), {}, []
    captured = []
    recipient_results = []

    def inference(core, kwargs, **unused):
        captured.append((core.session_id, copy.deepcopy(kwargs)))
        if core._delegate_depth:
            started.add(core.session_id)
            assert release.wait(15), "fixture child was not released"
            finished.add(core.session_id)
            return response("Correction: quantity is 2 cases; preserve the source hold.")
        sid = core.session_id
        turns[sid] = turns.get(sid, 0) + 1
        if sid == "fixture-caller":
            if turns[sid] == 1:
                return response(call=tool_call("a2a_call", {
                    "agent": base, "message": "Investigate the quantity discrepancy",
                    "context_id": "ctx-fixture-original",
                }, "caller-consult"))
            reply = next(m["content"] for m in reversed(kwargs["messages"]) if m["role"] == "tool")
            return response("Same caller continues: " + reply)
        if turns[sid] == 1:
            return response(call=tool_call("delegate_task", {"tasks": [
                {"goal": "Investigate the source quantity"},
                {"goal": "Independently check the source hold"}], "background": True}, "recipient-investigate"))
        if options.get("approval") and turns[sid] == 2:
            return response(call=tool_call("fixture_sensitive_action", {}, "protected-action"))
        if options.get("approval"):
            return response("Action denied; correction is 2 cases; source hold remains.")
        result = json.loads(next(m["content"] for m in reversed(kwargs["messages"]) if m["role"] == "tool"))
        recipient_results.append(result)
        if result.get("status") == "dispatched":
            return response("Investigation dispatched; hold unchanged.")
        return response(result["results"][0]["summary"])

    monkeypatch.setattr(AIAgent, "_interruptible_api_call", inference)
    monkeypatch.setattr(AIAgent, "_interruptible_streaming_api_call", inference)

    def core(sid):
        agent = AIAgent(
            api_key="fixture-key", base_url="https://example.invalid/v1", model="test-model",
            api_mode="chat_completions", session_id=sid, session_db=db,
            quiet_mode=True, skip_context_files=True, skip_memory=True,
            save_trajectories=False, enabled_toolsets=["delegation", "a2a"],
        )
        agent.compression_enabled = False
        return agent

    recipient, caller = core("fixture-recipient"), core("fixture-caller")

    async def handle(event):
        events.append(event)
        context = SessionContext(source=event.source, connected_platforms=[], home_channels={},
                                 session_key="fixture-recipient-key", session_id=recipient.session_id)
        with runner._session_env_scope(context):
            result = await runner._run_in_executor_with_context(
                lambda: recipient.run_conversation(event.text, task_id=event.message_id))
        return result["final_response"]

    adapter.set_message_handler(handle)
    assert await adapter.connect()
    base = f"http://127.0.0.1:{adapter._httpd.server_port}/"
    harness = SimpleNamespace(adapter=adapter, base=base, db=db, release=release,
                              started=started, finished=finished, events=events,
                              recipient=recipient, caller=caller, runner=runner,
                              captured=captured, recipient_results=recipient_results,
                              options=options, decisions=decisions, marker=marker)
    try:
        yield harness
    finally:
        release.set()
        await until(lambda: not adapter._session_tasks and not has_live_for_session(
            parent_session_id=recipient.session_id))
        await adapter.disconnect()
        recipient.close()
        caller.close()
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_push", [False, True])
async def test_native_a2a_waits_for_child_and_original_caller_continues(native_a2a, monkeypatch, legacy_push):
    h = native_a2a
    if legacy_push:
        # Reproduce the old inherited capability without changing the harness.
        monkeypatch.setattr(h.adapter, "supports_async_delivery", True)
    call = asyncio.create_task(asyncio.to_thread(
        h.caller.run_conversation, "Resolve this case through the peer", task_id="original-case"))
    try:
        await until(lambda: len(h.started) == 2 and len(h.events) == 1)
        event = h.events[0]
        task_id = event.message_id
        assert event.source.chat_id == "ctx-fixture-original"
        assert not h.finished
        if legacy_push:
            result = await asyncio.wait_for(asyncio.shield(call), 10)
            assert "Investigation dispatched; hold unchanged" in result["final_response"]
            assert "completed" in result["final_response"]
            assert not h.finished
        else:
            task = (await asyncio.to_thread(rpc, h.base, "tasks/get", {"taskId": task_id}))["result"]
            assert task["status"]["state"] == protocol.STATE_WORKING
            assert not call.done(), "dispatch ACK incorrectly ended the original caller"
    finally:
        h.release.set()
        result = await asyncio.wait_for(call, 20)
        await until(lambda: len(h.finished) == 2 and not h.adapter._session_tasks)
    saved = (await asyncio.to_thread(rpc, h.base, "tasks/get", {"taskId": task_id}))["result"]
    assert saved["id"] == task_id and saved["contextId"] == "ctx-fixture-original"
    assert saved["status"]["state"] == protocol.STATE_COMPLETED
    text = protocol.extract_text(saved["artifacts"][0])
    if legacy_push:
        # The late child cannot revise the completed Task or resume this caller.
        assert text == "Investigation dispatched; hold unchanged."
        assert "Correction:" not in result["final_response"]
    else:
        assert text == "Correction: quantity is 2 cases; preserve the source hold."
        assert "Same caller continues:" in result["final_response"] and text in result["final_response"]
        assert "status" not in h.recipient_results[0]
        assert all(r["status"] == "completed" for r in h.recipient_results[0]["results"])
    # Tool-call identity and the sent prefix survive the continuation.
    caller_requests = [kw for sid, kw in h.captured if sid == "fixture-caller"]
    assert caller_requests[1]["messages"][:len(caller_requests[0]["messages"])] == caller_requests[0]["messages"]
    history = h.db.get_messages_as_conversation("fixture-caller")
    assert any(m.get("tool_call_id") == "caller-consult" and text in m["content"] for m in history)


@pytest.mark.asyncio
async def test_a2a_cancel_stays_canceled_after_joined_child_finishes(native_a2a):
    h = native_a2a
    call = asyncio.create_task(asyncio.to_thread(tools.a2a_call, {
        "agent": h.base, "message": "Investigate", "context_id": "ctx-cancel"}))
    try:
        await until(lambda: len(h.started) == 2 and h.events)
        task_id = h.events[0].message_id
        result = await asyncio.to_thread(rpc, h.base, "tasks/cancel", {"taskId": task_id})
        assert result["result"]["status"]["state"] == protocol.STATE_CANCELED
        assert "canceled" in await asyncio.wait_for(asyncio.shield(call), 10)
    finally:
        h.release.set()
        await asyncio.wait_for(call, 20)
        await until(lambda: not h.adapter._session_tasks)
    task = (await asyncio.to_thread(rpc, h.base, "tasks/get", {"taskId": task_id}))["result"]
    assert task["status"]["state"] == protocol.STATE_CANCELED
    assert not task.get("artifacts")


@pytest.mark.asyncio
async def test_joined_a2a_keeps_actual_deny_and_returns_feedback(native_a2a, monkeypatch):
    from tools import approval

    h = native_a2a
    h.options["approval"] = True
    h.release.set()
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    pending = []
    key = "fixture-recipient-key"
    approval.register_gateway_notify(key, pending.append)
    call = asyncio.create_task(asyncio.to_thread(tools.a2a_call, {
        "agent": h.base, "message": "Investigate", "context_id": "ctx-deny"}))
    try:
        await until(lambda: bool(pending))
        assert not approval.is_session_yolo_enabled(key)
        assert approval.resolve_gateway_approval(key, "deny") == 1
        reply = await asyncio.wait_for(call, 15)
        assert "Action denied; correction is 2 cases; source hold remains." in reply
        assert h.decisions[0]["approved"] is False
        assert not h.marker.exists()
    finally:
        approval.unregister_gateway_notify(key)
        await asyncio.wait_for(call, 20)


@pytest.mark.asyncio
async def test_ordinary_native_chat_still_dispatches_without_waiting(native_a2a):
    h = native_a2a
    # Ordinary push adapters/unknown transports keep the existing True default.
    h.runner.adapters[Platform.LOCAL] = SimpleNamespace()
    context = SessionContext(source=SessionSource(platform=Platform.LOCAL, chat_id="fixture-chat"),
                             connected_platforms=[], home_channels={}, session_key="fixture-chat")
    with h.runner._session_env_scope(context):
        result = await h.runner._run_in_executor_with_context(
            lambda: h.recipient.run_conversation("Investigate from ordinary chat"))
    await until(lambda: len(h.started) == 2)
    assert not h.finished
    assert result["final_response"] == "Investigation dispatched; hold unchanged."
    assert h.recipient_results[0]["status"] == "dispatched"
    h.release.set()
    await until(lambda: len(h.finished) == 2)
