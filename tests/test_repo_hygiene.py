"""Tracked files say what a change protects, not which private ticket asked for it.

Comments, docstrings, test names and docs used to cite private tracker ids and
user-testing codes (the session, round, wave and batch labels of internal test
sweeps). A reader outside the project cannot look any of them up, so they read
as noise, and the sentence around them usually never said what the code
actually guards. These tests keep them from coming back: write the behaviour
instead ("a same-size edit within one second must still invalidate").

The patterns are assembled from pieces so this file does not match itself.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

# The nearest folder holding pyproject.toml, so the file can live at any depth.
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())

_ID = "CAS" + "-"
_SESSION = "r" + r"\d+" + "s" + r"\d+"

# Each pattern, with what it catches. Matching is case-insensitive.
PATTERNS = {
    "private tracker id": re.compile(r"\b" + _ID + r"\d+", re.IGNORECASE),
    # Not "--rounds 15", a benchmark's command-line flag.
    "test round": re.compile(r"(?<![-\w])rounds?[ -]?\d+", re.IGNORECASE),
    "test session code": re.compile(r"\b" + _SESSION + r"\b", re.IGNORECASE),
    "test wave": re.compile(r"\bwave[ -]?\d+", re.IGNORECASE),
    "test batch": re.compile(r"\bbatch[ -]?\d+", re.IGNORECASE),
    "numbered finding": re.compile(r"\bfinding #?\d+", re.IGNORECASE),
    "internal task": re.compile(r"\btask-\d+\b", re.IGNORECASE),
    "user-testing role": re.compile(r"\b" + "test" + r"ers?\b", re.IGNORECASE),
}

# A round code wrapped onto the next line of a comment or docstring, which the
# line-by-line check cannot see: "(round" at a line end, "28)" on the next.
WRAPPED_ROUND = re.compile(r"\bround\r?\n[ \t]*(?:#[ \t]*)?\d+\b", re.IGNORECASE)

# A file NAMED after a ticket ("test_" + "cas" + "123_....py").
PATH_PATTERN = re.compile(r"cas\d+", re.IGNORECASE)

# Files the content check skips. Keep this list short and explained.
SKIPPED_FILES = {
    # The changelog is history: it may name what shipped the way it was known then.
    "CHANGELOG.md",
}


def _tracked_files() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout; the tracked-file list is unavailable")
    return [p for p in out.decode("utf-8").split("\0") if p]


def _read_text(rel: str) -> str | None:
    path = REPO_ROOT / rel
    if not path.is_file():  # deleted in the working tree, or a symlink to a directory
        return None
    data = path.read_bytes()
    if b"\0" in data:  # binary
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _scanned_files() -> list[str]:
    return [rel for rel in _tracked_files() if rel not in SKIPPED_FILES]


def test_no_tracker_ids_or_test_round_codes_in_tracked_files():
    found = []
    for rel in _scanned_files():
        text = _read_text(rel)
        if text is None:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            for what, pattern in PATTERNS.items():
                m = pattern.search(line)
                if m:
                    found.append(f"{rel}:{n}: {what} {m.group(0)!r}: {line.strip()[:120]}")
        for m in WRAPPED_ROUND.finditer(text):
            n = text.count("\n", 0, m.start()) + 1
            found.append(f"{rel}:{n}: test round wrapped onto the next line {m.group(0)!r}")
    assert not found, (
        "Say what the code protects instead of citing a private ticket or a "
        "user-testing code (see this test's docstring):\n" + "\n".join(found)
    )


def test_no_file_is_named_after_a_ticket():
    named = [rel for rel in _scanned_files() if PATH_PATTERN.search(rel)]
    assert not named, "Name these files after the behaviour they check:\n" + "\n".join(named)


@pytest.mark.parametrize(
    "text",
    [
        _ID + "123",
        "round {n}",
        "Round-{n} gate",
        "Rounds {n}-18",
        "finding {n}",
        "pre-" + "Task" + "-5 path",
        "r{n}s{n}",
        "wave {n}",
        "Batch {n}",
        "finding #{n}",
        "a test" + "er said",
        "two test" + "ers",
    ],
)
def test_the_patterns_catch_what_they_are_for(text):
    """Without this, a broken pattern would pass the check above silently."""
    text = text.format(n=23)
    assert any(p.search(text) for p in PATTERNS.values())


@pytest.mark.parametrize(
    "text",
    [
        "cash.cache",
        "round(x, 2)",
        "test_round3_edge_cases.py",
        "test_tester_sessions.py",
        "batch_size = 32",
        "a test writer",
        "--rounds 15",
        "for r in range(rounds):",
        "a network round trip",
    ],
)
def test_the_patterns_leave_ordinary_text_alone(text):
    assert not any(p.search(text) for p in PATTERNS.values())


def test_a_round_code_wrapped_onto_the_next_line_is_caught():
    assert WRAPPED_ROUND.search("# came from a file (round\n        # 28), and")
    assert WRAPPED_ROUND.search("another cell -- round\n21), and")
    assert not WRAPPED_ROUND.search("x = round\n(y)")
