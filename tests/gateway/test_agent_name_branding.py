"""Owned display labels follow the rendering profile; payloads remain untouched."""
from pathlib import Path
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent import secret_scope
from agent.i18n import t
from gateway.config import PlatformConfig
from gateway.run import _format_exec_approval_fallback, _profile_runtime_scope
from hermes_cli.skin_engine import load_skin
from plugins.platforms.slack.adapter import SlackAdapter


@pytest.mark.parametrize("configured", [None, "", "   ", "Chase", "Ada"])
def test_branding_follows_real_profile_scope_without_rewriting_payloads(tmp_path, monkeypatch, configured):
    root = tmp_path / ".hermes"
    other = root / "profiles" / "other"
    other.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("AGENT_NAME", "Launch profile must not leak")
    (root / ".env").write_text("" if configured is None else f'AGENT_NAME="{configured}"\n')
    (other / ".env").write_text('AGENT_NAME="Other Agent"\n')
    for home in (root, other):
        (home / "config.yaml").write_text("display:\n  language: en\n")
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="test"))
    skin = load_skin("default")  # deliberately reuse one skin across A -> B -> A
    command = 'printf "Hermes"; hermes --help # HERMES_HOME ~/.hermes https://hermes-agent.nousresearch.com'
    reason = "User quoted Hermes and model NousResearch/Hermes"
    previous = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        for home, expected in ((root, (configured or "").strip() or "Hermes"),
                               (other, "Other Agent"),
                               (root, (configured or "").strip() or "Hermes")):
            with _profile_runtime_scope(home):
                card = adapter._format_exec_approval(command, reason)
                fallback = _format_exec_approval_fallback(command, reason, "/")
                for rendered in (card, fallback):
                    assert f"{expected} wants to run" in rendered
                    assert command in rendered
                    assert reason in rendered
                assert "`/approve`" in fallback and "`/deny`" in fallback
                assert adapter._EA_ACTION_IDS["once"] == "hermes_approve_once"
                for lang in ("en", "fr", "zh"):
                    assert expected in t("gateway.help.header", lang=lang)
                    assert expected in t("gateway.status.header", lang=lang)
                    resumed = t("gateway.resume.matrix_blocked_no_origin", lang=lang, name="Hermes")
                    assert "`/resume --cross-room Hermes`" in resumed
                assert expected in skin.get_branding("agent_name")
                assert expected in skin.get_branding("response_label")
                assert expected in skin.get_branding("welcome")
                assert "Hermes" in t("gateway.update.hermes_cmd_not_found", lang="en")
                from gateway.platforms.whatsapp_common import WhatsAppBehaviorMixin
                from gateway.relay.command_manifest import build_relay_command_manifest
                from hermes_cli.slack_cli import _build_full_manifest
                from plugins.platforms.email.adapter import EmailAdapter
                class WhatsAppProbe(WhatsAppBehaviorMixin):
                    _reply_prefix: str | None = None

                whatsapp = WhatsAppProbe()
                assert expected in whatsapp._effective_reply_prefix()
                whatsapp._reply_prefix = "Hermes custom prefix"
                assert whatsapp._effective_reply_prefix() == "Hermes custom prefix"
                manifest = _build_full_manifest(expected, "", messaging_experience="agent")
                assert expected in manifest["features"]["agent_view"]["agent_description"]
                descriptions = {row["name"]: row["description"] for row in build_relay_command_manifest()}
                assert expected in descriptions["status"]
                assert "Hermes" in descriptions["update"]  # actual software, not an agent label
                email = object.__new__(EmailAdapter)
                email._address = "sender@example.invalid"
                email._thread_context = {}
                message, message_id, subject = email._new_reply("recipient@example.invalid", reason)
                assert expected in subject and message_id.startswith("<hermes-")
                email._thread_context = {"recipient@example.invalid": {"subject": "Hermes user subject"}}
                assert email._new_reply("recipient@example.invalid", reason)[2] == "Re: Hermes user subject"
                client = AsyncMock()
                client.chat_postMessage.return_value = {"ts": "1.2"}
                client.api_call.return_value = {"ts": "1.3"}
                adapter._app = MagicMock()
                adapter._get_client = MagicMock(return_value=client)

                async def send_cards():
                    assert await adapter.create_handoff_thread("C1", "Hermes user title") == "1.2"
                    assert f"{expected} handoff" in client.chat_postMessage.call_args.kwargs["text"]
                    assert "Hermes user title" in client.chat_postMessage.call_args.kwargs["text"]
                    tasks = [{"id": "hermes_task", "title": "User's Hermes task", "status": "in_progress"}]
                    for title in (None, "Hermes custom title"):
                        result = await adapter.send_native_task_card_progress(
                            "C1", tasks, title=title,
                            metadata={"thread_id": "1.2", "team_id": "T1", "user_id": "U1"})
                        assert result.success
                        chunks = client.api_call.call_args.kwargs["json"]["chunks"]
                        assert chunks[0]["title"] == (title if title is not None else f"{expected} is working")
                        assert chunks[1]["title"] == tasks[0]["title"]
                        assert chunks[1]["id"] == tasks[0]["id"]

                asyncio.run(send_cards())
        # Unset does not inherit the launch env under multiplexing.
    finally:
        secret_scope.set_multiplex_active(previous)
