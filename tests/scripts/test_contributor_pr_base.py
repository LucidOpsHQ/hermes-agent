"""Execute attribution CI against the actual PR base, not a stale default branch."""
from pathlib import Path
import os
import subprocess

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.platforms("posix")


def git(repo, *args, email="fixture-new@example.invalid"):
    env = os.environ.copy()
    env.update(GIT_AUTHOR_NAME="Attribution fixture", GIT_COMMITTER_NAME="Attribution fixture",
               GIT_AUTHOR_EMAIL=email, GIT_COMMITTER_EMAIL=email)
    return subprocess.check_output(["git", *args], cwd=repo, env=env, text=True).strip()


def history(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "commit", "--allow-empty", "-qm", "old default")
    root = git(tmp_path, "rev-parse", "HEAD")
    git(tmp_path, "update-ref", "refs/remotes/origin/main", root)
    git(tmp_path, "commit", "--allow-empty", "-qm", "already in PR base", email="fixture-history@example.invalid")
    base = git(tmp_path, "rev-parse", "HEAD")
    git(tmp_path, "commit", "--allow-empty", "-qm", "new feature")
    directory = tmp_path / "contributors/emails"
    directory.mkdir(parents=True)
    (directory / "fixture-new@example.invalid").write_text("fixture-author\n")
    return base


def run_workflow(repo, base):
    workflow = yaml.safe_load((ROOT / ".github/workflows/contributor-check.yml").read_text())
    step = next(s for s in workflow["jobs"]["check-attribution"]["steps"] if s.get("id") == "check-emails")
    env = os.environ.copy()
    env["GITHUB_OUTPUT"] = str(repo / "github-output")
    for name, value in step.get("env", {}).items():
        env[name] = value.replace("${{ github.event.pull_request.base.sha }}", base)
    return subprocess.run(["bash", "-e", "-c", step["run"]], cwd=repo, env=env,
                          capture_output=True, text=True, timeout=15)


def test_nondefault_pr_base_does_not_claim_historical_authors(tmp_path):
    base = history(tmp_path)
    result = run_workflow(tmp_path, base)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "All contributor emails are mapped" in result.stdout
    assert "fixture-history@example.invalid" not in result.stdout


def test_unmapped_new_contributor_still_fails(tmp_path):
    base = history(tmp_path)
    (tmp_path / "contributors/emails/fixture-new@example.invalid").unlink()
    result = run_workflow(tmp_path, base)
    assert result.returncode != 0
    assert "fixture-new@example.invalid" in result.stdout


def test_invalid_pr_base_cannot_be_shell_input(tmp_path):
    history(tmp_path)
    result = run_workflow(tmp_path, "bad; touch escaped")
    assert result.returncode != 0
    assert not (tmp_path / "escaped").exists()
