"""Every ``noqa`` names its rule and says why the rule does not apply there.

Ruff checks the first half: PGH004 rejects a bare ``noqa`` and RUF100 rejects
one that silences nothing. It has no rule for the second half, the reason, so
this test holds it: ``# noqa: BLE001 - a notice must never break a call``.
"""

from __future__ import annotations

import io
import re
import subprocess
import tokenize
from pathlib import Path

import pytest

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())

# Ruff does not lint these, so a noqa in them is not a suppression.
EXCLUDED_PREFIXES = ("tests/docs/_fixtures/",)

NOQA = re.compile(r"#\s*noqa\b", re.IGNORECASE)
# `noqa: CODE[, CODE]` then a separator and some words of reason.
WITH_REASON = re.compile(r"#\s*noqa:\s*[A-Z]+[0-9]+(?:\s*,\s*[A-Z]+[0-9]+)*\s*(?:-|--|—)\s*\S")


def _tracked_python_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", "*.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout; the tracked-file list is unavailable")
    rels = [p for p in out.decode("utf-8").split("\0") if p and not p.startswith(EXCLUDED_PREFIXES)]
    return [REPO_ROOT / rel for rel in rels if (REPO_ROOT / rel).is_file()]


def _comments(source: str):
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            yield tok.start[0], tok.string


def _unexplained(source: str) -> list[tuple[int, str]]:
    return [(n, c) for n, c in _comments(source) if NOQA.search(c) and not WITH_REASON.search(c)]


def test_every_noqa_gives_a_reason():
    found = []
    for path in _tracked_python_files():
        source = path.read_text(encoding="utf-8")
        for n, comment in _unexplained(source):
            found.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{n}: {comment}")
    assert not found, "Say why the rule does not apply, as `# noqa: CODE - reason`:\n" + "\n".join(found)


@pytest.mark.parametrize(
    "line",
    [
        "x = 1  # noqa\n",
        "x = 1  # noqa: F401\n",
        "x = 1  # noqa: F401, E402\n",
        "x = 1  # noqa: BLE001 (why)\n",
    ],
)
def test_a_noqa_without_a_reason_is_caught(line):
    assert _unexplained(line)


@pytest.mark.parametrize(
    "line",
    [
        "x = 1  # noqa: F401 - an availability probe\n",
        "x = 1  # noqa: F401, E402 -- re-exported\n",
        "x = 1  # noqa: BLE001 — best effort\n",
        # Text that mentions noqa inside a string is not a suppression.
        's = "# noqa"\n',
    ],
)
def test_a_noqa_with_a_reason_passes(line):
    assert not _unexplained(line)
