"""CI must run the suite by EXCLUSION, never by enumeration.

The unit-test step used to name the handful of files it ran. That silently
omitted 18 top-level test files and four of the five test directories — and,
worse, anything added afterwards. The cost was concrete: a tracked regression
test for an urgent correctness bug (upstream simulation corrupting a
saved chart) sat red on ``main`` for weeks because CI never executed the file it
lived in. The suite was green and the gate was blind.

An allowlist rots invisibly, because the failure mode is a test that is never
run rather than a test that fails. These tests pin the inverted default: CI
targets ``tests/`` wholesale, and every exclusion must be spelled out where a
reviewer can see it.

The notebook integration suite is too slow for every push, so it is covered in
two ways instead: every push runs the core set (``tools/test_selection/``,
picked to cover every line, feature and step sequence the whole suite covers),
and a nightly workflow runs the whole folder in shards. The tests at the end
pin both.

This file is deliberately dependency-free — it parses the workflow as text
rather than importing a YAML library, so it cannot itself be skipped in an
environment that is missing something. The one exception collects the core set
with pytest itself, which CI installs for every job that runs this file.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

# The nearest folder holding pyproject.toml, so the file can live at any depth.
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
CI_YML = REPO_ROOT / ".github" / "workflows" / "ci.yml"
NIGHTLY_YML = REPO_ROOT / ".github" / "workflows" / "nightly.yml"
CORE_SET = REPO_ROOT / "tools" / "test_selection" / "core_set.txt"
PYPROJECT = REPO_ROOT / "pyproject.toml"
INTEGRATION = "tests/test_notebook_integration"
TESTS_DIR = REPO_ROOT / "tests"

# Directories under tests/ that the unit job legitimately does not run, each
# because another job covers them or they are too expensive per-push. Adding an
# entry here is a deliberate act that shows up in review; forgetting to add one
# makes test_every_test_directory_is_accounted_for fail.
EXPECTED_EXCLUSIONS = {
    "test_notebook_integration",  # starts real kernels; the core set and the nightly shards run it
    "test_wheel_gate",  # builds a wheel + real Jupyter server; run by hand before a release
    "docs",  # dedicated docs-parity job (needs docs-test extras)
}


@pytest.fixture(scope="module")
def unit_step() -> str:
    """The shell body of the 'Run unit tests' step."""
    assert CI_YML.is_file(), f"missing workflow: {CI_YML}"
    text = CI_YML.read_text(encoding="utf-8")
    m = re.search(
        r"- name: Run unit tests\s*\n\s*run: \|(?P<body>.*?)(?=\n\s*- name:|\n\n\s*#|\n  [a-z-]+:)",
        text,
        re.DOTALL,
    )
    if m is None:
        # Fall back to "everything from the step header to the next '- name:'".
        start = text.index("- name: Run unit tests")
        rest = text[start + 1 :]
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
            "blind spot this test guards.\nStep body was:\n" + unit_step
        )

    def test_unit_step_does_not_enumerate_individual_files(self, unit_step):
        """A stray `tests/test_foo.py` argument means someone re-narrowed it."""
        enumerated = re.findall(r"tests/[\w\-/]*\.py", unit_step)
        assert not enumerated, (
            "The unit-test step names individual test files: "
            f"{enumerated}. Exclude what must not run with --ignore instead; "
            "enumeration silently drops everything not listed."
        )


class TestStepsParseOnEveryRunner:
    """A step that runs on Windows must parse in the shell Windows uses.

    Windows runners default to ``pwsh``, where a trailing ``\\`` is not a line
    continuation. A multi-line command written in bash style dies with
    "Missing expression after unary operator '--'" BEFORE pytest starts, so the
    job fails with no test output at all — it reads like a test failure and
    isn't one. Every Windows job failed this way for two full runs.

    Any step whose body relies on backslash continuations therefore has to
    declare ``shell: bash`` (available on all three runners).
    """

    def test_backslash_continuations_declare_bash(self):
        text = CI_YML.read_text(encoding="utf-8")
        steps = re.split(r"\n(?=\s*- name: )", text)
        offenders = []
        for step in steps:
            m = re.search(r"- name: (?P<name>.+)", step)
            if m is None:
                continue
            body = step.split("run:", 1)[1] if "run:" in step else ""
            uses_continuation = re.search(r"\\\s*\n", body)
            declares_bash = re.search(r"shell:\s*bash", step)
            if uses_continuation and not declares_bash:
                offenders.append(m.group("name").strip())
        assert not offenders, (
            "These CI steps use `\\` line continuations without `shell: bash`. "
            "They parse in bash but not in pwsh, so they break on the Windows "
            f"runners before running anything: {offenders}"
        )


class TestExclusionsAreHonest:
    def test_every_exclusion_still_exists(self, unit_step):
        """A stale --ignore hides that its target vanished or was renamed."""
        for name in _ignored_paths(unit_step):
            assert (TESTS_DIR / name).exists(), (
                f"ci.yml ignores tests/{name}, which no longer exists. Remove the stale --ignore."
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

        This is the test that would have caught the unrun file: it fails the moment a
        directory exists that CI neither runs nor names.
        """
        on_disk = {p.name for p in TESTS_DIR.iterdir() if p.is_dir() and not p.name.startswith("__")}
        holding_tests = {name for name in on_disk if any((TESTS_DIR / name).rglob("test_*.py"))}
        unaccounted = on_disk - _ignored_paths(unit_step)
        # One direction: nothing is ignored that isn't a real directory of tests.
        assert _ignored_paths(unit_step) <= holding_tests, (
            f"ci.yml ignores directories that do not exist or hold no tests: "
            f"{sorted(_ignored_paths(unit_step) - holding_tests)}"
        )
        # The other: every directory of tests the step does not ignore is run.
        # The step targets tests/, so only pytest's own configuration could
        # still hide one from it.
        pytest_config = PYPROJECT.read_text(encoding="utf-8").split("[tool.pytest.ini_options]", 1)[1]
        pytest_config = pytest_config.split("\n[", 1)[0]
        hiding = [
            re.search(r"norecursedirs|--ignore|--deselect", pytest_config) and "pyproject.toml",
            *(
                str(c.relative_to(REPO_ROOT))
                for c in TESTS_DIR.rglob("conftest.py")
                if re.search(r"collect_ignore|pytest_ignore_collect", c.read_text(encoding="utf-8"))
            ),
        ]
        hiding = [h for h in hiding if h]
        assert not hiding, (
            f"{hiding} can hide test directories from `pytest tests/`, so the unit "
            "step may not run everything it does not ignore. Exclude in ci.yml instead."
        )
        # Sanity: the directories we expect to be covered really are.
        for name in ("test_core", "test_backends", "test_ui", "test_notebook"):
            assert name in unaccounted, f"tests/{name} is not being run by CI — it is excluded or gone."


def _step_run(text: str, name: str) -> str:
    """The ``run:`` body of the step called *name*, up to the next step."""
    start = text.index(f"- name: {name}")
    nxt = text.find("- name:", start + 1)
    step = text[start:] if nxt == -1 else text[start:nxt]
    assert "run:" in step, f"step {name!r} has no run:"
    return step.split("run:", 1)[1]


class TestTheIntegrationSuiteRuns:
    def test_every_push_runs_the_core_set(self):
        run = _step_run(CI_YML.read_text(encoding="utf-8"), "Run the integration core set")
        assert re.search(r"pytest\s+@tools/test_selection/core_set\.txt", run), run

    def test_ci_names_no_integration_file_by_hand(self):
        """The core set is the list; a hand-picked one beside it goes stale."""
        named = re.findall(INTEGRATION + r"/[\w\-/]*\.py", CI_YML.read_text(encoding="utf-8"))
        assert not named, f"ci.yml names integration test files by hand: {named}"

    def test_every_core_set_entry_is_an_integration_test_that_exists(self):
        """A renamed or deleted test must not leave a dead node id behind."""
        entries = [line.strip() for line in CORE_SET.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert entries, f"{CORE_SET} is empty"
        missing = [e for e in entries if not (REPO_ROOT / e.split("::")[0]).is_file()]
        outside = [e for e in entries if not e.startswith(INTEGRATION + "/")]
        assert not missing, "core_set.txt names test files that do not exist:\n" + "\n".join(missing)
        assert not outside, "core_set.txt names tests outside the integration suite:\n" + "\n".join(outside)

    @pytest.mark.timeout(180)
    def test_every_core_set_entry_is_a_test_pytest_collects(self):
        """pytest exits 4 without running anything when one `@file` entry names
        no test, so a renamed test or parametrize id fails every integration-core
        job. Checking only the file part misses that; collecting does not."""
        entries = {line.strip() for line in CORE_SET.read_text(encoding="utf-8").splitlines() if line.strip()}
        env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "-n",
                "0",
                "-p",
                "no:randomly",
                "-p",
                "no:cacheprovider",
                "@" + str(CORE_SET.relative_to(REPO_ROOT)),
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=170,
        )
        # One node id per line, then a blank line and the summary. Ids may hold spaces.
        collected = {line for line in proc.stdout.splitlines() if line.startswith("tests/")}
        # A missing node id makes pytest stop with "ERROR: not found: <id>".
        assert proc.returncode == 0, (
            f"pytest cannot collect core_set.txt (exit {proc.returncode}); "
            "rename the entry with the test or re-pick the core set.\n" + proc.stdout[-3000:] + proc.stderr[-3000:]
        )
        assert collected == entries, (
            f"entries pytest did not collect: {sorted(entries - collected)}\n"
            f"collected but not listed: {sorted(collected - entries)}"
        )

    def test_the_nightly_workflow_is_scheduled_and_can_be_run_by_hand(self):
        text = NIGHTLY_YML.read_text(encoding="utf-8")
        assert re.search(r"^\s*schedule:\s*\n\s*- cron: \"[^\"]+\"", text, re.MULTILINE), text
        assert re.search(r"^\s*workflow_dispatch:", text, re.MULTILINE), text

    def test_the_nightly_workflow_runs_the_whole_folder_in_shards(self):
        run = _step_run(NIGHTLY_YML.read_text(encoding="utf-8"), "Run integration shard")
        assert re.search(r"pytest\s+" + INTEGRATION + r"\s", run), run
        assert "-p tools.test_selection.shard" in run
        assert "--shard=${{ matrix.shard }}/${{ strategy.job-total }}" in run
        pytest_args = run.split("pytest", 1)[1]
        for narrowing in ("--ignore", "--deselect", " -k ", " -m ", "::"):
            assert narrowing not in pytest_args, f"the nightly run is narrowed by {narrowing.strip()!r}: {run}"

    def test_the_nightly_matrix_is_the_shard_numbers_one_to_n(self):
        """N is the job count, so any other matrix axis would skip shards."""
        text = NIGHTLY_YML.read_text(encoding="utf-8")
        m = re.search(r"^(?P<indent>\s*)matrix:\s*\n(?P<body>(?:(?P=indent)\s+.*\n)+)", text, re.MULTILINE)
        assert m, "no matrix in nightly.yml"
        keys = re.findall(r"^\s*([\w-]+):", m.group("body"), re.MULTILINE)
        assert keys == ["shard"], f"the nightly matrix must hold only `shard`, found {keys}"
        values = [int(v) for v in re.search(r"shard:\s*\[([^\]]+)\]", m.group("body")).group(1).split(",")]
        assert values == list(range(1, len(values) + 1)), values
