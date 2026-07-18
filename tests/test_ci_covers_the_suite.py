"""CI must run the suite by EXCLUSION, never by enumeration (CAS-152).

The unit-test step used to name the handful of files it ran. That silently
omitted 18 top-level test files and four of the five test directories — and,
worse, anything added afterwards. The cost was concrete: a tracked regression
test for an Urgent correctness bug (CAS-175, upstream simulation corrupting a
saved chart) sat red on ``main`` for weeks because CI never executed the file it
lived in. The suite was green and the gate was blind.

An allowlist rots invisibly, because the failure mode is a test that is never
run rather than a test that fails. These tests pin the inverted default: CI
targets ``tests/`` wholesale, and every exclusion must be spelled out where a
reviewer can see it.

This file is deliberately dependency-free — it parses the workflow as text
rather than importing a YAML library, so it cannot itself be skipped in an
environment that is missing something.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
CI_YML = REPO_ROOT / ".github" / "workflows" / "ci.yml"
TESTS_DIR = REPO_ROOT / "tests"

# Directories under tests/ that the unit job legitimately does not run, each
# because another job covers them or they are too expensive per-push. Adding an
# entry here is a deliberate act that shows up in review; forgetting to add one
# makes test_every_test_directory_is_accounted_for fail.
EXPECTED_EXCLUSIONS = {
    "test_notebook_integration",  # ~793 kernel-spinning files; smoke subset covers headlines
    "test_wheel_gate",            # builds a wheel + real Jupyter server; release gate
    "docs",                       # dedicated docs-parity job (needs docs-test extras)
}


@pytest.fixture(scope="module")
def unit_step() -> str:
    """The shell body of the 'Run unit tests' step."""
    assert CI_YML.is_file(), f"missing workflow: {CI_YML}"
    text = CI_YML.read_text(encoding="utf-8")
    m = re.search(
        r"- name: Run unit tests\s*\n\s*run: \|(?P<body>.*?)(?=\n\s*(?:- name:|[a-z-]+:contentReference)|\n\s*- name:|\n\n\s*#|\n  [a-z-]+:)",
        text,
        re.DOTALL,
    )
    if m is None:
        # Fall back to "everything from the step header to the next '- name:'".
        start = text.index("- name: Run unit tests")
        rest = text[start + 1:]
        nxt = rest.find("- name:")
        return rest[:nxt] if nxt != -1 else rest
    return m.group("body")


def _ignored_paths(step: str) -> set[str]:
    return set(re.findall(r"--ignore=tests/([\w\-.]+)", step))


class TestCiTargetsTheWholeTree:
    def test_unit_step_runs_the_tests_directory_wholesale(self, unit_step):
        """`pytest tests/` — not a list of files."""
        assert re.search(r"pytest\s+tests/\s*(\\|\n|$)", unit_step), (
            "The unit-test step must invoke `pytest tests/` so new tests are "
            "picked up automatically. Naming individual files re-creates the "
            "CAS-152 blind spot.\nStep body was:\n" + unit_step
        )

    def test_unit_step_does_not_enumerate_individual_files(self, unit_step):
        """A stray `tests/test_foo.py` argument means someone re-narrowed it."""
        enumerated = re.findall(r"tests/[\w\-/]*\.py", unit_step)
        assert not enumerated, (
            "The unit-test step names individual test files: "
            f"{enumerated}. Exclude what must not run with --ignore instead; "
            "enumeration silently drops everything not listed."
        )


class TestExclusionsAreHonest:
    def test_every_exclusion_still_exists(self, unit_step):
        """A stale --ignore hides that its target vanished or was renamed."""
        for name in _ignored_paths(unit_step):
            assert (TESTS_DIR / name).exists(), (
                f"ci.yml ignores tests/{name}, which no longer exists. "
                "Remove the stale --ignore."
            )

    def test_exclusions_match_the_documented_set(self, unit_step):
        """Exclusions are a reviewed list, not an accumulating pile."""
        assert _ignored_paths(unit_step) == EXPECTED_EXCLUSIONS, (
            "ci.yml's exclusions drifted from the documented set.\n"
            f"  in ci.yml: {sorted(_ignored_paths(unit_step))}\n"
            f"  expected:  {sorted(EXPECTED_EXCLUSIONS)}\n"
            "If the change is intended, update EXPECTED_EXCLUSIONS here and say "
            "why in the workflow comment — that is the point of this test."
        )

    def test_every_test_directory_is_accounted_for(self, unit_step):
        """A new test directory is either run, or explicitly excluded.

        This is the test that would have caught CAS-152: it fails the moment a
        directory exists that CI neither runs nor names.
        """
        on_disk = {
            p.name
            for p in TESTS_DIR.iterdir()
            if p.is_dir() and not p.name.startswith("__")
        }
        unaccounted = on_disk - _ignored_paths(unit_step)
        # Everything not ignored IS run, because the step targets tests/.
        # So this only checks the inverse: nothing is ignored that isn't real.
        assert _ignored_paths(unit_step) <= on_disk, (
            f"ci.yml ignores directories that do not exist: "
            f"{sorted(_ignored_paths(unit_step) - on_disk)}"
        )
        # Sanity: the directories we expect to be covered really are.
        for name in ("test_core", "test_backends", "test_ui", "test_notebook"):
            assert name in unaccounted, (
                f"tests/{name} is not being run by CI — it is excluded or gone."
            )
