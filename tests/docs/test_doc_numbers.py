"""The numbers the docs quote are derived, not remembered.

Every hand-maintained figure in the documentation drifted at once and nothing
noticed: the README said ~8,500 tests while ``testing.md`` said ~8,750 and,
three sections later, ~8,800; the claim count said 164 against an actual 247;
``versioning.md`` pinned ``~=0.2.0`` six minor versions after the fact. Prose
is not executed, ``mkdocs --strict`` only checks links, and the claim verifier
only checks that pinned *symbols* still exist -- so a wrong number renders
exactly as well as a right one.

``scripts/doc_numbers.py`` re-derives each figure from the repository and
rewrites it between markers. This test is the gate that makes the derivation
load-bearing rather than optional.

**The split.** This test runs ``--fast``: version, claim count, platform count
and test-file count, all of which are cheap file reads. The test *counts*
require three ``pytest --collect-only`` passes (~15 s) and would be recursive
here, so CI runs the full ``--check`` as its own step in the docs job. Anything
that can be checked without spawning pytest is checked on every ordinary run of
this suite.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "doc_numbers.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO, capture_output=True, text=True,
    )


def test_the_cheap_derived_numbers_are_current():
    """The gate: a stale version or claim count fails the build."""
    result = _run("--check", "--fast")
    assert result.returncode == 0, (
        "a number quoted in the docs no longer matches the repository.\n"
        "Run `python scripts/doc_numbers.py --update`.\n\n"
        + result.stdout + result.stderr
    )


def test_every_marker_names_a_fact_that_exists():
    """A typo'd marker never updates and would otherwise never complain.

    ``--check`` reports unknown marker names before it compares values, so this
    is covered by the test above; it is asserted separately because "the
    marker silently does nothing" is the precise failure mode the whole
    mechanism exists to remove, and a regression there would look like a pass.
    """
    result = _run("--check", "--fast")
    assert "Unknown docnum markers" not in result.stdout, result.stdout


def test_the_checker_actually_fails_on_drift(tmp_path):
    """A gate that cannot fail is a green checkmark with nothing behind it.

    Rather than mutate the real docs, this drives the script's own comparison
    against a corrupted copy of one marked file.
    """
    target = REPO / "docs" / "how-it-works" / "testing.md"
    original = target.read_text(encoding="utf-8")
    assert "<!-- docnum:platforms -->" in original, (
        "testing.md lost its platform-count marker; the derivation is no "
        "longer covering the number it was added for"
    )

    backup = tmp_path / "testing.md.bak"
    backup.write_text(original, encoding="utf-8")
    corrupted = original.replace(
        "<!-- docnum:platforms -->", "<!-- docnum:platforms -->999", 1
    )
    assert corrupted != original

    try:
        target.write_text(corrupted, encoding="utf-8")
        result = _run("--check", "--fast")
        assert result.returncode == 1, (
            "the checker passed against a deliberately wrong number:\n"
            + result.stdout
        )
        assert "docnum:platforms" in result.stdout, result.stdout
    finally:
        target.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")

    # And the restore worked, so the suite does not leave the repo dirty.
    assert target.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    "path, name",
    [
        ("README.md", "tests_total"),
        ("README.md", "platforms"),
        ("docs/how-it-works/testing.md", "claims"),
        ("docs/versioning.md", "version_pin"),
        ("docs/cli.md", "version"),
    ],
)
def test_the_numbers_that_drifted_are_the_ones_now_derived(path, name):
    """Pin the specific figures this mechanism was built for.

    Without this, someone could delete a marker (restoring the hand-maintained
    number) and every other test here would still pass -- the checker only
    validates the markers that exist.
    """
    text = (REPO / path).read_text(encoding="utf-8")
    assert f"<!-- docnum:{name} -->" in text, (
        f"{path} no longer derives `{name}`; it is hand-maintained again"
    )
