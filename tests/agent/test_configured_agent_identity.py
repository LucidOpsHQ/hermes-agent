"""LO-326: shared identity follows the real profile scope, without rewriting SOUL."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import secret_scope
from agent.prompt_builder import DEFAULT_AGENT_IDENTITY
from agent.system_prompt import build_system_prompt_parts
from gateway.run import _profile_runtime_scope
from hermes_cli.config import ensure_hermes_home


def _agent(*, soul_state, tools):
    return SimpleNamespace(
        load_soul_identity=soul_state != "skipped", skip_context_files=True,
        valid_tool_names=tools, _task_completion_guidance=False,
        _parallel_tool_call_guidance=False, _tool_use_enforcement=False,
        _execution_guidance=False, _environment_probe=False,
        _bot_mode_protocol=False, _kanban_worker_guidance="",
        _memory_store=None, _memory_manager=None, model="test-model",
        provider="test-provider", platform="cli", pass_session_id=False,
        session_id="20261029_120000_identity", _session_db=None,
    )


@pytest.mark.parametrize("other_name", ["Beta", "", "   "])
@pytest.mark.parametrize("soul_state", ["present", "missing", "skipped"])
@pytest.mark.parametrize("skills", ["no-tools", "no-skill", "installed"])
def test_identity_in_real_profile_prompt_assembly(tmp_path, monkeypatch, other_name, soul_state, skills):
    response_style = DEFAULT_AGENT_IDENTITY.partition(". ")[2]
    root = tmp_path / ".hermes"
    other = root / "profiles" / "other"
    other.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("AGENT_NAME", "Launch identity must not leak")
    monkeypatch.chdir(tmp_path)
    (root / ".env").write_text('AGENT_NAME="Alpha"\n')
    (other / ".env").write_text(f'AGENT_NAME="{other_name}"\n' if other_name else "")
    personas = {}
    previous = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        for home in (root, other):
            (home / "config.yaml").write_text("display:\n  language: en\n")
            # Initialize the synthetic home before removing its seeded SOUL;
            # missing-file coverage must exercise the fallback, not first-run seeding.
            with _profile_runtime_scope(home):
                ensure_hermes_home()
            persona = f"Profile {home.name} persona: You are Hermes Agent, built by Nous Research. Keep replies brief."
            personas[home] = persona
            soul = home / "SOUL.md"
            if soul_state == "missing":
                soul.unlink()
            else:
                soul.write_text(persona)
            if skills == "installed":
                skill = home / "skills" / "hermes-agent" / "SKILL.md"
                skill.parent.mkdir(parents=True)
                skill.write_text("---\nname: hermes-agent\ndescription: Runtime help\n---\nHelp body.\n")
        tools = [] if skills == "no-tools" else ["skill_view"]
        prompts = []
        for home, expected in ((root, "Alpha"), (other, other_name.strip()), (root, "Alpha")):
            with _profile_runtime_scope(home):
                parts = build_system_prompt_parts(_agent(soul_state=soul_state, tools=tools))
            stable = parts["stable"]
            prompts.append(parts)
            assert "Launch identity must not leak" not in stable
            assert ("built by LucidOps" in stable) == bool(expected)
            if expected:
                assert f"Your name is {expected}, an AI agent built by LucidOps." in stable
                assert "ordinary identity or creator questions" in stable
                assert "explicit technical questions" in stable
                assert "Do not volunteer that stack or vendor introduction" in stable
                assert "even if SOUL.md names the runtime or its authors" in stable
                if soul_state != "present":
                    assert "You are Hermes Agent, built by Nous Research." not in stable
            if soul_state == "present":
                assert stable.startswith(personas[home])
                assert stable.count(personas[home]) == 1
                assert response_style not in stable
            else:
                assert personas[home] not in stable
                assert response_style in stable
                if not expected:
                    assert stable.startswith(DEFAULT_AGENT_IDENTITY)
            assert "https://hermes-agent.nousresearch.com/docs" in stable
            assert ("skill_view(name='hermes-agent')" in stable) == (skills == "installed")
            assert "Model: test-model" in parts["volatile"]
            assert "Provider: test-provider" in parts["volatile"]
            if soul_state == "missing":
                assert not (home / "SOUL.md").exists()
            else:
                assert (home / "SOUL.md").read_text() == personas[home]
        assert prompts[0] == prompts[2]  # A -> B -> A never freezes B's identity into A.
        assert "Your name is Alpha" not in prompts[1]["stable"]
        assert "Your name is Beta" not in prompts[0]["stable"]
    finally:
        secret_scope.set_multiplex_active(previous)


def test_identity_resolution_does_not_borrow_unscoped_launch_name(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("AGENT_NAME", "Launch identity must not leak")
    previous = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope(None)
    try:
        with pytest.raises(secret_scope.UnscopedSecretError):
            build_system_prompt_parts(_agent(soul_state="skipped", tools=[]))
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(previous)
