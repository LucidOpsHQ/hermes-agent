"""Lifecycle and topic labels use the owning profile without rewriting user content."""
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent import secret_scope
from gateway.config import Platform
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.run_topics import GatewayTopicThreadsMixin, _collapse_title
from gateway.session import SessionSource


@pytest.fixture
def homes(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    other = root / "profiles" / "other"
    other.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("AGENT_NAME", "Wrong launch name")
    (root / ".env").write_text("AGENT_NAME=Chase\n")
    (other / ".env").write_text("")
    for home in (root, other):
        (home / "config.yaml").write_text("display:\n  language: en\n")
    previous = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        yield root, other
    finally:
        secret_scope.set_multiplex_active(previous)


@pytest.mark.asyncio
async def test_shutdown_resolves_each_owner_without_ambient_scope(homes):
    runner = object.__new__(GatewayRunner)
    runner.config = MagicMock(multiplex_profiles=True)
    runner.config.get_home_channel.return_value = None
    runner._restart_requested = False
    runner._snapshot_running_agents = lambda: ("a", "b", "a2")
    sources = [SessionSource(platform=Platform.TELEGRAM, chat_id=str(i), profile=profile)
               for i, profile in enumerate(("default", "other", "default"))]
    runner._shutdown_notification_target = AsyncMock(side_effect=[
        (source, "telegram", source.chat_id, None, source.profile) for source in sources])
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=MagicMock(success=True))
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._adapter_for_source = lambda source: adapter
    runner._notice_allowed = lambda *args: True
    runner._thread_metadata_for_target = lambda *args, **kwargs: None
    await runner._notify_active_sessions_of_shutdown()
    messages = [call.args[1] for call in adapter.send.call_args_list]
    assert len(messages) == len(sources)
    for message, expected in zip(messages, ("Chase", "Hermes", "Chase")):
        assert f"⚠️ {expected} is shutting down" in message
        assert "Wrong launch name" not in message


@pytest.mark.asyncio
async def test_topic_rendering_follows_scope_and_preserves_restored_content(homes):
    runner = GatewayTopicThreadsMixin()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="1", user_id="2", thread_id="3")
    runner._is_telegram_topic_lane = lambda source: True
    runner._session_key_for_source = lambda source: "telegram:1:3"
    db = AsyncMock()
    runner._session_db = db
    db.resolve_session_id.return_value = "session-id"
    db.get_session.return_value = {"source": "telegram", "user_id": "2"}
    db.is_telegram_session_linked_to_topic.return_value = False
    db.get_session_title.return_value = "Hermes {user title}"
    db.get_messages.return_value = [{"role": "assistant", "content": "Hermes {unchanged response}"}]
    root, other = homes
    for home, expected in ((root, "Chase"), (other, "Hermes"), (root, "Chase")):
        with _profile_runtime_scope(home):
            assert _collapse_title("") == f"{expected} Chat"
            assert _collapse_title("Hermes user title") == "Hermes user title"
            assert f"independent {expected} session" in runner._telegram_topic_root_lobby_message()
            assert f"parallel {expected} chat" in runner._telegram_topic_root_new_message()
            assert f"new {expected} session" in runner._telegram_topic_new_header(source)
            assert f"{expected} checks BotFather" in runner._telegram_topic_help_text()
            restored = await runner._restore_telegram_topic_session(MagicMock(source=source), "session-id")
            assert f"Last {expected} message:" in restored
            assert "Hermes {user title}" in restored
            assert "Hermes {unchanged response}" in restored
