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

REPO_ROOT = Path(__file__).resolve().parent.parent

_ID = "CAS" + "-"
_SESSION = "r" + r"\d+" + "s" + r"\d+"

# Each pattern, with what it catches. Matching is case-insensitive.
PATTERNS = {
    "private tracker id": re.compile(r"\b" + _ID + r"\d+", re.IGNORECASE),
    "test round": re.compile(r"\bround[ -]?\d+", re.IGNORECASE),
    "test session code": re.compile(r"\b" + _SESSION + r"\b", re.IGNORECASE),
    "test wave": re.compile(r"\bwave[ -]?\d+", re.IGNORECASE),
    "test batch": re.compile(r"\bbatch[ -]?\d+", re.IGNORECASE),
    "numbered finding": re.compile(r"\bfinding #\d+", re.IGNORECASE),
    "user-testing role": re.compile(r"\b" + "test" + r"ers?\b", re.IGNORECASE),
}

# A file NAMED after a ticket ("test_" + "cas" + "123_....py").
PATH_PATTERN = re.compile(r"cas\d+", re.IGNORECASE)

# Paths the content check skips entirely. Keep this list short and explained.
SKIPPED_PREFIXES = (
    # TEMPORARY: src/ still carries tracker ids in comments and is being cleaned
    # up separately. Delete this entry once `src/` is clean; the check then
    # covers the whole repository.
    "src/",
)
SKIPPED_FILES = {
    # The changelog is history: it may name what shipped the way it was known then.
    "CHANGELOG.md",
}

# In this file only the named section is skipped: it explains how to map an old
# tracker id to its current issue, which needs the id format.
GUIDE = ".github/copilot-instructions.md"
GUIDE_SECTION = "## Project management"


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


def _skipped_line_numbers(rel: str, lines: list[str]) -> set[int]:
    """Line numbers (1-based) inside the guide's allowed section."""
    if rel != GUIDE:
        return set()
    skipped: set[int] = set()
    inside = False
    for n, line in enumerate(lines, 1):
        if line.startswith("## "):
            inside = line.strip() == GUIDE_SECTION
        if inside:
            skipped.add(n)
    return skipped


def _scanned_files() -> list[str]:
    return [rel for rel in _tracked_files() if not rel.startswith(SKIPPED_PREFIXES) and rel not in SKIPPED_FILES]


def test_no_tracker_ids_or_test_round_codes_in_tracked_files():
    found = []
    for rel in _scanned_files():
        text = _read_text(rel)
        if text is None:
            continue
        lines = text.splitlines()
        skipped = _skipped_line_numbers(rel, lines)
        for n, line in enumerate(lines, 1):
            if n in skipped:
                continue
            for what, pattern in PATTERNS.items():
                m = pattern.search(line)
                if m:
                    found.append(f"{rel}:{n}: {what} {m.group(0)!r}: {line.strip()[:120]}")
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
    ],
)
def test_the_patterns_leave_ordinary_text_alone(text):
    assert not any(p.search(text) for p in PATTERNS.values())


def test_the_allowed_guide_section_still_exists():
    """If the heading is renamed, the carve-out must be updated, not widened."""
    text = _read_text(GUIDE)
    assert text is not None
    assert GUIDE_SECTION in text.splitlines()
