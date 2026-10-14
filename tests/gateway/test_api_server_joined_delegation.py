"""Request-local joined delegation: real HTTP, session store, and native dispatcher.

Inference/remote-client construction is replaced in the dispatcher tests, not
child construction, child lifecycle, dispatch, aggregation, or context binding.
"""

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.session_context import get_session_env, session_history_delivery_supported
from hermes_constants import get_hermes_home
from hermes_state import SessionDB


async def until(predicate):
    async with asyncio.timeout(15):
        while not predicate():
            await asyncio.sleep(0.01)


class ProbeAgent:
    session_prompt_tokens = session_completion_tokens = session_total_tokens = 0

    def __init__(self, session_id, action):
        self.session_id = session_id
        self.action = action
        self.interrupted = threading.Event()

    def run_conversation(self, **kwargs):
        return self.action(**kwargs)

    def interrupt(self, *args, **kwargs):
        self.interrupted.set()


@pytest_asyncio.fixture
async def api(monkeypatch):
    home = get_hermes_home()
    (home / "config.yaml").write_text(
        "approvals:\n  mode: manual\n  timeout: 5\n"
        "security:\n  tirith_enabled: false\n"
        "delegation:\n  max_concurrent_children: 3\n  worktree: false\n",
        encoding="utf-8",
    )
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "joined-test-bearer"}))
    db = SessionDB(home / "state.db")
    adapter._session_db = db
    actions, seen = {}, []

    def create(**kwargs):
        sid = kwargs["session_id"]
        seen.append(kwargs)
        return ProbeAgent(sid, actions.get(sid, lambda **kw: {"final_response": "done"}))

    monkeypatch.setattr(adapter, "_create_agent", create)
    app = web.Application()
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    app.router.add_post("/v1/runs/{run_id}/stop", adapter._handle_stop_run)
    app.router.add_post("/v1/runs/{run_id}/approval", adapter._handle_run_approval)
    async with TestClient(TestServer(app), headers={"Authorization": "Bearer joined-test-bearer"}) as client:
        yield SimpleNamespace(client=client, adapter=adapter, db=db, actions=actions, seen=seen, home=home)
        tasks = list(adapter._active_run_tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await until(lambda: not adapter._active_run_workers)
    adapter._run_idempotency_store.close()
    db.close()


async def start(api, *, headers=None, **body):
    response = await api.client.post("/v1/runs", json={"input": "probe", **body}, headers=headers)
    assert response.status == 202, await response.text()
    return (await response.json())["run_id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, True, False, 0, 1, [], {}, ["joined"], "", "async", "JOINED", " joined"])
async def test_invalid_delegation_mode_is_rejected_before_admission(api, value):
    response = await api.client.post("/v1/runs", json={"input": "probe", "delegation_mode": value})
    assert response.status == 400
    assert (await response.json())["error"]["code"] == "invalid_delegation_mode"
    assert not api.seen and not api.adapter._run_statuses and not api.adapter._active_run_tasks


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [None, "joined"])
@pytest.mark.parametrize("address", ["session", "header", "fresh", "caller", "chain"])
async def test_joined_mode_preserves_history_selection_and_compression(api, monkeypatch, mode, address):
    history = [
        {"role": "user", "content": "compressed context"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call-1", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"fixture"}'}}
        ]},
        {"role": "tool", "content": "native tool result", "tool_call_id": "call-1", "name": "read_file"},
        {"role": "assistant", "content": "retained reply", "reasoning_content": "retained reasoning"},
    ]
    api.db.create_session("parent", source="api_server", session_key="joined-history-key")
    api.db.append_message("parent", "user", "stale pre-compression detail")
    api.db.publish_compression_child(
        parent_session_id="parent", child_session_id="tip", source="api_server",
        messages=history, require_compression_lease=False,
    )
    snapshot = [{"role": "user", "content": "caller snapshot"}]
    api.adapter._response_store.put("resp-seed", {"conversation_history": snapshot, "session_id": "parent"})
    observed = {}

    def probe(**kwargs):
        observed.update(kwargs)
        observed["chat_id"] = get_session_env("HERMES_SESSION_CHAT_ID")
        observed["delivery"] = session_history_delivery_supported()
        return {"final_response": "done"}

    monkeypatch.setattr(api.adapter, "_create_agent", lambda **kw: ProbeAgent(kw["session_id"], probe))
    options = {
        "session": {"session_id": "parent"}, "header": {}, "fresh": {},
        "caller": {"session_id": "parent", "conversation_history": snapshot},
        "chain": {"previous_response_id": "resp-seed"},
    }[address]
    headers = {"X-Hermes-Session-Key": "joined-history-key"} if address == "header" else None
    run_id = await start(api, headers=headers, **options, **({"delegation_mode": mode} if mode else {}))
    await until(lambda: run_id not in api.adapter._active_run_tasks)
    assert api.adapter._run_statuses[run_id]["status"] == "completed"
    assert observed["chat_id"] == (run_id if address == "fresh" else "tip")
    assert observed["delivery"] is (mode is None and address not in {"caller", "chain"})
    native_history = api.db.get_messages_as_conversation("tip")
    assert native_history[1]["tool_calls"] == history[1]["tool_calls"]
    assert native_history[2]["tool_call_id"] == "call-1"
    assert native_history[3]["reasoning_content"] == "retained reasoning"
    expected = snapshot if address in {"caller", "chain"} else [] if address == "fresh" else native_history
    assert observed["conversation_history"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [None, "joined"])
async def test_capability_mode_and_idempotency_are_durable(api, mode):
    capabilities = await (await api.client.get("/v1/capabilities")).json()
    assert capabilities["features"]["run_scoped_joined_delegation"] is True
    body = {"session_id": "durable", **({"delegation_mode": mode} if mode else {})}
    headers = {"Idempotency-Key": "joined-idempotency"}
    run_id = await start(api, headers=headers, **body)
    await until(lambda: run_id not in api.adapter._active_run_tasks)
    response = await api.client.post("/v1/runs", json={"input": "probe", **body}, headers=headers)
    assert response.status == 202 and response.headers["Idempotency-Replayed"] == "true"
    assert (await response.json())["run_id"] == run_id
    changed = {"input": "probe", "session_id": "durable", **({"delegation_mode": "joined"} if mode is None else {})}
    conflict = await api.client.post("/v1/runs", json=changed, headers=headers)
    assert conflict.status == 409
    assert (await conflict.json())["error"]["code"] == "idempotency_key_conflict"
    api.adapter._run_statuses.pop(run_id)
    status = await (await api.client.get(f"/v1/runs/{run_id}")).json()
    assert status["status"] == "completed"
    assert status["delegation_mode"] == (mode or "default")
    assert len(api.seen) == 1


@pytest.mark.asyncio
async def test_joined_mode_refuses_live_owner_without_forwarding(api):
    from hermes_cli.active_sessions import try_acquire_active_session
    from tools.bot_live_delivery import find_canonical_live_owner

    api.db.create_session("bot-chat", "desktop")
    api.db.set_session_title("bot-chat", "Bot Chat")
    lease, refusal = try_acquire_active_session(
        session_id="bot-chat", surface="desktop", config={}, registry_home=api.home, track_liveness=True,
        metadata={"live_session_id": "live-1", "bot_live_delivery_consumer": True},
    )
    assert lease is not None and refusal is None
    run_id = None
    try:
        assert find_canonical_live_owner(api.home) is not None
        run_id = await start(api, session_id="bot-chat", delegation_mode="joined")
        await until(lambda: api.adapter._run_statuses[run_id]["status"] != "queued")
        status = api.adapter._run_statuses[run_id]
        assert status["status"] == "failed"
        assert "live Bot Chat" in status["error"]
        assert not api.seen
        assert not list((api.home / "runtime" / "bot_live_delivery").glob("*.json"))
    finally:
        if run_id in api.adapter._active_run_tasks:
            await api.client.post(f"/v1/runs/{run_id}/stop")
            await until(lambda: run_id not in api.adapter._active_run_tasks)
        lease.release()


@pytest.fixture
def native_core(monkeypatch, tmp_path):
    """Real parent/child AIAgents; no remote clients, discovery, or inference."""
    from run_agent import AIAgent
    from tools.delegate_tool import DELEGATE_TASK_SCHEMA

    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **kw: None)
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", MagicMock())
    monkeypatch.setattr("agent.model_metadata.fetch_model_metadata", lambda *a, **kw: {})
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *a, **kw: [
        {"type": "function", "function": DELEGATE_TASK_SCHEMA}])
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda *a, **kw: {})
    parents = []

    def create(session_id, db):
        core = AIAgent(
            api_key="test-key", base_url="https://example.invalid/v1", model="test-model",
            session_id=session_id, session_db=db, quiet_mode=True, skip_context_files=True,
            skip_memory=True, platform="api_server", enabled_toolsets=["delegation"],
        )
        parents.append(core)
        return core

    yield create
    for core in parents:
        core.close()


@pytest.mark.asyncio
async def test_native_dispatch_joins_children_without_changing_concurrent_default(api, native_core, monkeypatch, tmp_path):
    from run_agent import AIAgent
    from tools.async_delegation import has_live_for_session
    from tools.process_registry import process_registry

    release = threading.Event()
    started, finished, tool_results = {}, {}, {}
    lock = threading.Lock()
    cores = {name: native_core(name, api.db) for name in ("joined", "default")}
    for name in cores:
        api.db.create_session(name, "api_server")
        started[name], finished[name] = set(), set()

    def child_inference(child, **kwargs):
        parent = child._delegate_parent_ref()
        name, child_id = parent.session_id, child._subagent_id
        with lock:
            started[name].add(child_id)
        assert release.wait(15)
        summary = f"{name}/{child_id} result"
        (tmp_path / child_id).write_text(summary, encoding="utf-8")
        with lock:
            finished[name].add(summary)
        return {"final_response": summary, "completed": True, "messages": [], "api_calls": 0}

    monkeypatch.setattr(AIAgent, "run_conversation", child_inference)

    def create(**kwargs):
        core = cores[kwargs["session_id"]]
        core.tool_progress_callback = kwargs["tool_progress_callback"]

        def parent_turn(**turn):
            call = SimpleNamespace(id=f"call-{core.session_id}", type="function", function=SimpleNamespace(
                name="delegate_task", arguments=json.dumps({"tasks": [
                    {"goal": "Write the first fixture result"}, {"goal": "Write the second fixture result"}]})))
            messages = []
            core._execute_tool_calls(SimpleNamespace(tool_calls=[call]), messages, turn["task_id"])
            tool_results[core.session_id] = json.loads(messages[-1]["content"])
            return {"final_response": messages[-1]["content"]}

        monkeypatch.setattr(core, "run_conversation", parent_turn)
        return core

    monkeypatch.setattr(api.adapter, "_create_agent", create)
    run_ids = {}
    try:
        run_ids["joined"] = await start(api, session_id="joined", delegation_mode="joined")
        run_ids["default"] = await start(api, session_id="default")
        await until(lambda: all(len(v) == 2 for v in started.values()))
        await until(lambda: run_ids["default"] not in api.adapter._active_run_tasks)
        assert tool_results["default"]["status"] == "dispatched"
        assert api.adapter._run_statuses[run_ids["joined"]]["status"] == "running"
        assert "joined" not in tool_results
        assert not any(finished.values())
    finally:
        release.set()
        await until(lambda: all(run_id not in api.adapter._active_run_tasks for run_id in run_ids.values()))
        await until(lambda: not any(has_live_for_session(parent_session_id=name) for name in cores))
    result = tool_results["joined"]
    assert "status" not in result  # a joined result, never a detached handle
    assert [entry["task_index"] for entry in result["results"]] == [0, 1]
    assert all(entry["status"] == "completed" for entry in result["results"])
    assert {entry["summary"] for entry in result["results"]} == finished["joined"]
    assert all((tmp_path / child_id).exists() for ids in started.values() for child_id in ids)
    events = []
    while not process_registry.completion_queue.empty():
        events.append(process_registry.completion_queue.get_nowait())
    detached = [event for event in events if event.get("type") == "async_delegation"]
    assert len(detached) == 1 and detached[0]["origin_session_id"] == "default"
    response = await api.client.get(f"/v1/runs/{run_ids['joined']}/events")
    stream = await response.text()
    assert stream.count('"event": "subagent.complete"') == 2
    assert '"event": "run.completed"' in stream
    assert stream.rfind('"event": "subagent.complete"') < stream.index('"event": "run.completed"')


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["deny", "stop", "cancel"])
async def test_joined_mode_keeps_native_human_approval_and_cancellation(api, finish, monkeypatch):
    from tools import approval

    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    decisions = []

    def probe(**kwargs):
        decisions.append(approval.request_tool_approval("joined-test", "approve test action"))
        return {"final_response": json.dumps(decisions[-1]), "interrupted": finish != "deny"}

    api.actions["approval-test"] = probe
    run_id = await start(api, session_id="approval-test", delegation_mode="joined")
    await until(lambda: api.adapter._run_statuses[run_id]["status"] == "waiting_for_approval")
    assert not approval.is_session_yolo_enabled(run_id)
    if finish == "deny":
        response = await api.client.post(f"/v1/runs/{run_id}/approval", json={"choice": "deny"})
        assert response.status == 200
    elif finish == "stop":
        response = await api.client.post(f"/v1/runs/{run_id}/stop")
        assert response.status == 200
    else:
        task = api.adapter._active_run_tasks[run_id]
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    await until(lambda: run_id not in api.adapter._active_run_tasks)
    assert decisions and decisions[0]["approved"] is False
    assert api.adapter._run_statuses[run_id]["status"] == ("completed" if finish == "deny" else "cancelled")
    assert run_id not in approval._gateway_notify_cbs
    assert run_id not in approval._gateway_queues
