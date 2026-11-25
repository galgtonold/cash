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

import json
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
PUBLISH_YML = REPO_ROOT / ".github" / "workflows" / "publish.yml"
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
    "test_wheel_gate",  # builds a wheel + real Jupyter server; the nightly workflow runs it
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


def _tiers() -> dict[str, dict[str, object]]:
    """The plan job's two tiers: ``{"full": {...}, "branch": {...}}``, each
    holding the ``unit`` matrix and the ``core`` Python list as parsed JSON."""
    text = CI_YML.read_text(encoding="utf-8")
    full, branch = text.split('if [ "$FULL_TIER" = true ]; then', 1)[1].split("\n          else\n", 1)
    branch = branch.split("\n          fi\n", 1)[0]
    tiers = {}
    for name, body in (("full", full), ("branch", branch)):
        tiers[name] = {
            key: json.loads(re.search(rf"\b{key}='(.*?)'", body, re.DOTALL).group(1)) for key in ("unit", "core")
        }
    return tiers


def _supported_pythons() -> list[str]:
    return re.findall(r'"Programming Language :: Python :: (3\.\d+)"', PYPROJECT.read_text(encoding="utf-8"))


class TestBothTiersRunTheSuite:
    """Working-branch pushes run a smaller matrix, with the unit suite split
    into shards; main, pull requests and runs by hand run the full one."""

    def test_the_unit_step_runs_the_matrix_shard(self, unit_step):
        assert "-p tools.test_selection.shard --shard=${{ matrix.shard }}" in unit_step, unit_step
        assert re.search(r"python -m pytest\s+tests/", unit_step), "-p tools... needs the repo root on sys.path"

    def test_both_tiers_take_their_matrix_from_the_plan(self):
        text = CI_YML.read_text(encoding="utf-8")
        assert "matrix: ${{ fromJSON(needs.plan.outputs.unit) }}" in _job(text, "test")
        assert "python-version: ${{ fromJSON(needs.plan.outputs.core) }}" in _job(text, "integration-core")
        cond = re.search(r"FULL_TIER: \$\{\{ (.+) \}\}", text).group(1)
        assert "github.event_name != 'push'" in cond and "refs/heads/main" in cond, cond

    def test_the_full_tier_runs_every_os_and_python_whole(self):
        full = _tiers()["full"]
        assert full["unit"]["os"] == ["ubuntu-latest", "windows-latest", "macos-latest"]
        assert full["unit"]["python-version"] == _supported_pythons()
        assert full["unit"]["shard"] == ["1/1"]
        assert full["core"] == _supported_pythons()

    def test_branch_shards_run_every_test_once_per_os_and_python(self):
        """The shards of one OS and Python are 1/N..N/N, so the shard plugin
        runs each test in exactly one of them."""
        groups: dict[tuple[str, str], list[str]] = {}
        for entry in _tiers()["branch"]["unit"]["include"]:
            assert set(entry) == {"os", "python-version", "shard"}, entry
            groups.setdefault((entry["os"], entry["python-version"]), []).append(entry["shard"])
        for key, shards in groups.items():
            n = len(shards)
            assert sorted(shards) == [f"{i}/{n}" for i in range(1, n + 1)], (key, shards)
        oldest, newest = _supported_pythons()[0], _supported_pythons()[-1]
        # The oldest Python on macOS, the newest on Windows and macOS: Windows
        # on the oldest, the slowest job of all, runs in the full tier only.
        assert {("macos-latest", oldest), ("macos-latest", newest), ("windows-latest", newest)} <= set(groups), groups
        assert any(os_name == "ubuntu-latest" for os_name, _ in groups), groups
        assert {oldest, newest} <= set(_tiers()["branch"]["core"])

    def test_the_branch_tier_fits_the_free_concurrency_limits(self):
        """20 concurrent jobs, 5 of them on macOS, so a push never queues."""
        text = CI_YML.read_text(encoding="utf-8")
        branch = _tiers()["branch"]
        jobs = re.findall(r"^  ([\w-]+):\n", text.split("\njobs:\n", 1)[1], re.MULTILINE)
        single = [j for j in jobs if j not in ("test", "integration-core")]
        total = len(single) + len(branch["unit"]["include"]) + len(branch["core"])
        macos = sum(e["os"].startswith("macos") for e in branch["unit"]["include"])
        assert total <= 20, total
        assert macos <= 5, macos


def _step(text: str, name: str) -> str:
    """The step called *name*, up to the next step."""
    start = text.index(f"- name: {name}")
    nxt = text.find("- name:", start + 1)
    return text[start:] if nxt == -1 else text[start:nxt]


def _step_run(text: str, name: str) -> str:
    """The ``run:`` body of the step called *name*."""
    step = _step(text, name)
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
        step = _step(NIGHTLY_YML.read_text(encoding="utf-8"), "Run integration shard")
        run = step.split("run:", 1)[1]
        assert re.search(r"pytest\s+" + INTEGRATION + r"\s", run), run
        assert "-p tools.test_selection.shard" in run
        assert "--shard=${{ matrix.shard }}/$SHARDS" in run
        pytest_args = run.split("pytest", 1)[1]
        for narrowing in ("--ignore", "--deselect", " -k ", " -m ", "::"):
            assert narrowing not in pytest_args, f"the nightly run is narrowed by {narrowing.strip()!r}: {run}"

    def test_the_nightly_shards_are_one_to_n_and_n_is_what_the_step_divides_by(self):
        """A shard number missing from the matrix, or a divisor other than
        their count, leaves tests that no job runs."""
        text = NIGHTLY_YML.read_text(encoding="utf-8")
        job = _job(text, "integration")
        m = re.search(r"^(?P<indent>\s*)matrix:\s*\n(?P<body>(?:(?P=indent)\s+.*\n)+)", job, re.MULTILINE)
        assert m, "no matrix in the integration job"
        keys = re.findall(r"^\s*([\w-]+):", m.group("body"), re.MULTILINE)
        assert keys == ["python", "shard"], f"the nightly matrix holds `python` and `shard`, found {keys}"
        values = [int(v) for v in re.search(r"shard:\s*\[([^\]]+)\]", m.group("body")).group(1).split(",")]
        assert values == list(range(1, len(values) + 1)), values
        step = _step(text, "Run integration shard")
        assert re.search(r'SHARDS: "(\d+)"', step).group(1) == str(len(values)), step
        # The job total counts the Pythons too, so it cannot be the divisor.
        assert "strategy.job-total" not in job

    def test_the_nightly_shards_run_on_the_oldest_and_the_newest_python(self):
        """3.10 has no C extension and no run-time trace: another engine keys
        the code and watches the body there."""
        job = _job(NIGHTLY_YML.read_text(encoding="utf-8"), "integration")
        pythons = re.findall(r'"(3\.\d+)"', re.search(r"python: \[([^\]]+)\]", job).group(1))
        supported = _supported_pythons()
        assert {supported[0], supported[-1]} <= set(pythons), (pythons, supported)
        assert "python-version: ${{ matrix.python }}" in job

    def test_a_scheduled_run_tests_the_merge_branch(self):
        """A schedule starts from the default branch, which a change reaches
        last: the whole suite would only ever run on what is already main."""
        job = _job(NIGHTLY_YML.read_text(encoding="utf-8"), "integration")
        checkout = job.index("- name: Check out the merge branch")
        assert checkout < job.index("- name: Install dependencies")
        step = job[checkout : job.index("- name: Set up Python")]
        # By hand, the branch the run was started on is the one tested.
        assert "if: github.event_name == 'schedule'" in step
        assert "claude/batch" in step and "git checkout" in step


class TestWorkflowsAreLockedDown:
    """An action named by a tag or a branch is whatever that name points at
    on the day of the run, with the workflow's token in its hands; a commit
    hash is one reviewed version. The token itself reads the repository and
    no more, unless a job asks."""

    WORKFLOWS = sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml"))

    @pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda p: p.name)
    def test_every_action_is_pinned_to_a_commit(self, workflow):
        used = re.findall(r"^\s*(?:- )?uses:\s*(\S+)", workflow.read_text(encoding="utf-8"), re.MULTILINE)
        assert used, f"{workflow.name} uses no action: the check would pass on nothing"
        loose = [u for u in used if not u.startswith("./") and not re.fullmatch(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}", u)]
        assert not loose, f"{workflow.name}: pin by commit hash, with the version in a comment: {loose}"

    @pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda p: p.name)
    def test_the_token_only_reads_unless_a_job_asks_for_more(self, workflow):
        text = workflow.read_text(encoding="utf-8")
        assert re.search(r"^permissions:\n  contents: read\n(?=\S|\n)", text, re.MULTILINE), workflow.name


class TestTheSpeedSetIsReported:
    def test_the_nightly_workflow_compares_the_speed_set_with_main(self):
        job = _job(NIGHTLY_YML.read_text(encoding="utf-8"), "speed")
        assert "benchmarks/speed_compare.py origin/main HEAD" in job
        assert (REPO_ROOT / "benchmarks" / "speed_compare.py").is_file()
        # Both refs must be in the checkout to be compared.
        assert "fetch-depth: 0" in job

    def test_a_slower_scenario_warns_and_does_not_fail_the_run(self):
        """A shared runner's timings are too noisy to gate on."""
        job = _job(NIGHTLY_YML.read_text(encoding="utf-8"), "speed")
        assert "continue-on-error: true" in job
        assert "::warning::" in job


class TestTheHeldTestsRunNightly:
    """A push skips the tests marked ``nightly`` and runs a few seeds of each
    ``full_sweep`` sweep (`tests._sweeps`); the nightly held-unit job runs
    both in full, so nothing the unit job leaves out is left out everywhere."""

    def test_the_unit_step_deselects_only_perf_and_nightly(self, unit_step):
        marks = re.findall(r'-m "([^"]*)"', unit_step)
        assert marks == ["not perf and not nightly"], marks

    def test_the_nightly_workflow_runs_them_over_the_unit_tree(self):
        job = _job(NIGHTLY_YML.read_text(encoding="utf-8"), "held-unit")
        assert 'CASH_TEST_FULL_SWEEP: "1"' in job, job
        run = _step_run(job, "Run the held unit tests")
        assert re.search(r"pytest\s+tests/\s", run), run
        assert set(re.findall(r"--ignore=tests/([\w\-.]+)", run)) == EXPECTED_EXCLUSIONS, run
        assert re.findall(r'-m "([^"]*)"', run) == ["nightly or full_sweep"], run
        for narrowing in ("--deselect", " -k ", "::", "--shard"):
            assert narrowing not in run, f"the held-unit run is narrowed by {narrowing.strip()!r}: {run}"

    def test_every_sweep_is_marked_for_the_nightly_job(self):
        """A module that shortens its sweep per push must be one the nightly
        job selects, or its other seeds never run anywhere."""
        sweeping = [
            p
            for p in TESTS_DIR.rglob("test_*.py")
            if re.search(r"^from tests\._sweeps import|^import tests\._sweeps", p.read_text(encoding="utf-8"), re.M)
        ]
        assert sweeping, "no module sweeps through tests._sweeps; the check has nothing to check"
        unmarked = [
            str(p.relative_to(REPO_ROOT))
            for p in sweeping
            if "pytest.mark.full_sweep" not in p.read_text(encoding="utf-8")
        ]
        assert not unmarked, f"these modules shorten a sweep per push but are not marked full_sweep: {unmarked}"


class TestTheSelfChecksRunNightly:
    """``CASH_VERIFY=1`` turns on every memo's check against the slow path
    (`cash._verify`). No push runs with it (a checked run is several times
    slower); the nightly self-check job runs the unit suite and the
    integration core set that way, so a memo that disagrees with the slow
    path fails a job the morning after."""

    def test_the_nightly_workflow_runs_the_unit_suite_and_the_core_set_checked(self):
        job = _job(NIGHTLY_YML.read_text(encoding="utf-8"), "self-check")
        unit = _step_run(job, "Run the unit tests with every self-check on")
        assert re.search(r"pytest\s+tests/\s", unit), unit
        ignored = set(re.findall(r"--ignore=tests/([\w\-./]+)", unit))
        # The work-count tests bound the work a call does; a check adds to it.
        assert ignored == EXPECTED_EXCLUSIONS | {"test_core/internals", "test_notebook/internals"}, unit
        assert re.findall(r'-m "([^"]*)"', unit) == ["not perf and not nightly"], unit
        for narrowing in ("--deselect", " -k ", "::", "--shard"):
            assert narrowing not in unit, f"the self-check run is narrowed by {narrowing.strip()!r}: {unit}"
        core = _step_run(job, "Run the integration core set with every self-check on")
        assert "@tools/test_selection/core_set.txt" in core, core
        assert job.count('CASH_VERIFY: "1"') == 2, job

    def test_no_push_runs_with_the_self_checks_on(self):
        assert "CASH_VERIFY:" not in CI_YML.read_text(encoding="utf-8")

    def test_the_switch_covers_every_check(self):
        """A check added with a switch of its own is asked for through
        `cash._verify.requested`, or the one switch leaves it off."""
        src = REPO_ROOT / "src" / "cash"
        own = set()
        direct = []
        for path in src.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            own |= set(re.findall(r'_verify\.requested\("(CASH_[A-Z_]+)"\)', text))
            direct += [
                f"{path.relative_to(src)}: {m}"
                for m in re.findall(r'(?:environ\.get|getenv|_flag)\(\s*"(CASH_\w*VERIFY\w*)"', text)
                if not m.endswith("_LOG")
            ]
        assert {"CASH_HITPATH_VERIFY", "CASH_STATE_MEMO_VERIFY", "CASH_CLASS_MEMO_VERIFY"} <= own, own
        assert {"CASH_ANALYSIS_MEMO_VERIFY", "CASH_CWATCH_VERIFY"} <= own, own
        assert not direct, f"read from the environment past cash._verify: {direct}"


def _job(text: str, name: str) -> str:
    """The body of the top-level job called *name*, up to the next job."""
    m = re.search(rf"^  {re.escape(name)}:\n(?P<body>(?:(?:    .*|\s*)\n)+)", text + "\n", re.MULTILINE)
    assert m, f"no job {name!r}"
    return m.group("body")


class TestTheWheelIsChecked:
    """The suites run the editable install; only these jobs see the built wheel."""

    def test_the_nightly_workflow_runs_the_wheel_gate(self):
        job = _job(NIGHTLY_YML.read_text(encoding="utf-8"), "wheel-gate")
        assert 'CASH_WHEEL_GATE: "1"' in job, job
        assert re.search(r"pytest\s+-m wheel_gate tests/test_wheel_gate", job), job

    def test_nothing_is_uploaded_before_the_wheel_is_verified(self):
        text = PUBLISH_YML.read_text(encoding="utf-8")
        for name in ("publish", "publish-test"):
            needs = re.search(r"^    needs: (.+)$", _job(text, name), re.MULTILINE)
            assert needs and "verify-wheel" in needs.group(1), f"{name} can upload an unverified wheel"

    def test_a_release_uploads_the_platform_wheels_and_the_pure_wheel(self):
        """The platform wheels come from cibuildwheel (wheels.yml), which
        runs the C extension's tests on each one; the build job adds the universal
        pure-Python wheel, which verify-wheel checks."""
        text = PUBLISH_YML.read_text(encoding="utf-8")
        assert "uses: ./.github/workflows/wheels.yml" in _job(text, "wheels")
        assert "CASH_PURE_PYTHON=1 python -m build --wheel" in _job(text, "build")
        for name in ("publish", "publish-test"):
            job = _job(text, name)
            assert re.search(r"^    needs: .*\bwheels\b", job, re.MULTILINE), (
                f"{name} can run before the wheels are built"
            )
            assert "pattern: cibw-wheels-*" in job, f"{name} does not upload the platform wheels"

    def test_the_verification_checks_version_and_extension(self):
        job = _job(PUBLISH_YML.read_text(encoding="utf-8"), "verify-wheel")
        assert "python -m venv" in job
        assert "GITHUB_REF_NAME#v" in job and "cash.__version__" in job
        assert "labextension list" in job and "cash-live-cells" in job

    def test_every_platform_wheel_runs_the_c_extension_tests(self):
        """Importing ``cash._cwatch`` passed a Windows-only bug (a 32-bit
        ``long``); each wheel runs the watcher tests, the fuzz and a
        verify-mode smoke on its own platform, against the installed wheel."""
        config = PYPROJECT.read_text(encoding="utf-8").split("[tool.cibuildwheel]", 1)[1].split("\n[", 1)[0]
        assert re.search(r'^test-command = "python -m tests\._wheel_selftest"$', config, re.MULTILINE), config
        sources = re.search(r"^test-sources = \[(.*)\]$", config, re.MULTILINE)
        assert sources and {'"tests"', '"pyproject.toml"'} <= {s.strip() for s in sources.group(1).split(",")}, config
        selftest = (TESTS_DIR / "_wheel_selftest.py").read_text(encoding="utf-8")
        for runs in ('"tests/test_core/watchers"', "tests.test_core.watchers._fuzz", '"CASH_CWATCH_VERIFY": "1"'):
            assert runs in selftest, f"tests/_wheel_selftest.py no longer runs {runs}"


class TestTheOldestPythonInstallsFromWheels:
    """cash's oldest Python hangs on its dependencies still releasing for it.
    pip says nothing when one stops (it installs the last release that
    does), so a nightly job installs cash from wheels only on that Python
    and reports a dependency that fell behind."""

    def test_the_nightly_job_runs_on_the_oldest_supported_python(self):
        job = _job(NIGHTLY_YML.read_text(encoding="utf-8"), "oldest-python-install")
        assert f'python-version: "{_supported_pythons()[0]}"' in job, job

    def test_it_installs_wheels_only_and_reports(self):
        job = _job(NIGHTLY_YML.read_text(encoding="utf-8"), "oldest-python-install")
        install = _step_run(job, "Install it from wheels only")
        assert "--only-binary=:all:" in install, install
        assert "b.ALGORITHM == 'blake3'" in install, install
        assert "tools/dependency_floor.py" in _step_run(job, "Report dependencies that left this Python behind")
        assert (REPO_ROOT / "tools" / "dependency_floor.py").is_file()


class TestTheLibraryCodeReadIsChecked:
    """A notebook rule that rests on a library's pickling code is off, with
    no word, on a release that edits that code. A nightly job on the newest
    releases is where a maintainer hears of it."""

    def test_the_nightly_job_runs_the_check_on_the_newest_releases(self):
        job = _job(NIGHTLY_YML.read_text(encoding="utf-8"), "library-code")
        assert "pip install --upgrade numpy scikit-learn" in job
        assert "python tools/library_code_read.py" in job
        assert (REPO_ROOT / "tools" / "library_code_read.py").is_file()
        # It fails the run: the fix is a maintainer reading the new code.
        assert "continue-on-error" not in job
