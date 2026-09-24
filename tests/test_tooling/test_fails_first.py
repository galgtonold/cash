"""scripts/fails_first.py runs the new tests against the last commit's source.

It used to ``git stash push -- src/``, run the tests, then ``git stash pop``.
``git stash push`` leaves untracked files alone, so a fix that only added a
file under src/ stashed nothing: the "unfixed" run tested the fixed source, and
the pop then applied an older, unrelated stash of the user's. Its exit code
also said "real guard" when only one of several tests failed, and when the
tests could not even be collected.

These tests drive the real script in a scratch git repository.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
SCRIPT = REPO_ROOT / "scripts" / "fails_first.py"

# Reads the answer from a data file when one ships next to it. The fix in the
# first test adds only that file, so it is untracked and nothing else changes.
MOD = """\
from pathlib import Path

_DATA = Path(__file__).with_name("answer.txt")


def answer():
    return int(_DATA.read_text(encoding="utf-8")) if _DATA.exists() else 41
"""

GUARD = """\
from pkg.mod import answer


def test_answer_is_42():
    assert answer() == 42
"""

VACUOUS = """\
def test_true():
    assert True
"""


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
    return proc.stdout


def _write(repo: Path, rel: str, text: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A committed package whose ``answer()`` is wrong, plus the script."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "commit.gpgsign", "false")
    _write(repo, "src/pkg/__init__.py", "")
    _write(repo, "src/pkg/mod.py", MOD)
    _write(repo, "notes.txt", "committed\n")
    (repo / "scripts").mkdir()
    shutil.copy2(SCRIPT, repo / "scripts" / "fails_first.py")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "initial")
    return repo


def _fails_first(repo: Path, *targets: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(repo / "scripts" / "fails_first.py"), *targets],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _state(repo: Path) -> tuple[str, str, str]:
    return (
        _git(repo, "status", "--porcelain", "--untracked-files=all"),
        _git(repo, "stash", "list"),
        _git(repo, "worktree", "list"),
    )


def test_an_untracked_fix_is_left_out_and_an_old_stash_is_left_alone(repo):
    # An unrelated stash the user made earlier.
    _write(repo, "notes.txt", "stashed work\n")
    _git(repo, "stash", "push", "--quiet", "-m", "unrelated")
    # The fix is one new, untracked file under src/; the new test is untracked too.
    _write(repo, "src/pkg/answer.txt", "42")
    _write(repo, "tests/test_answer.py", GUARD)
    before = _state(repo)

    proc = _fails_first(repo, "tests/test_answer.py")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "FAILED" in proc.stdout and "test_answer_is_42" in proc.stdout, proc.stdout
    assert _state(repo) == before, "the script changed the tree, the stashes or the worktrees"
    assert (repo / "notes.txt").read_text(encoding="utf-8") == "committed\n"
    assert (repo / "src/pkg/answer.txt").read_text(encoding="utf-8") == "42"


def test_a_staged_fix_and_a_changed_test_file_are_kept_as_they_were(repo):
    _write(repo, "tests/test_answer.py", VACUOUS)
    _git(repo, "add", "tests/test_answer.py")
    _git(repo, "commit", "--quiet", "-m", "placeholder test")
    _write(repo, "src/pkg/mod.py", MOD.replace("else 41", "else 42"))
    _git(repo, "add", "src/pkg/mod.py")
    _write(repo, "tests/test_answer.py", GUARD)
    before = _state(repo)

    proc = _fails_first(repo, "tests/test_answer.py")

    # The edited test file is what ran, not the committed placeholder.
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "test_answer_is_42" in proc.stdout, proc.stdout
    assert _state(repo) == before


def test_one_test_that_passes_without_the_fix_is_named_and_fails_the_check(repo):
    _write(repo, "src/pkg/mod.py", MOD.replace("else 41", "else 42"))
    _write(repo, "tests/test_answer.py", GUARD + "\n\n" + VACUOUS)

    proc = _fails_first(repo, "tests/test_answer.py")

    assert proc.returncode == 1, proc.stdout + proc.stderr
    vacuous = proc.stdout.split("VACUOUS", 1)[1]
    assert "test_true" in vacuous and "test_answer_is_42" not in vacuous, proc.stdout


def test_a_collection_error_is_inconclusive_not_a_pass(repo):
    _write(repo, "src/pkg/mod.py", MOD + "\n\nNEW = 1\n")
    _write(repo, "tests/test_answer.py", "from pkg.mod import NEW\n\n\ndef test_new():\n    assert NEW == 1\n")

    proc = _fails_first(repo, "tests/test_answer.py")

    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "INCONCLUSIVE" in proc.stdout, proc.stdout


def test_without_a_source_change_there_is_nothing_to_compare(repo):
    _write(repo, "tests/test_answer.py", GUARD)

    proc = _fails_first(repo, "tests/test_answer.py")

    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "no uncommitted changes under src/" in proc.stdout, proc.stdout
