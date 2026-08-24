"""The mutation harness must reach the kernel -- proven, not assumed.

A mutation harness that silently fails to apply is worse than none: it reports
that the suite tolerates a broken engine, when in fact the engine was never
broken. That mistake has already been made here once, patching
``_backward_scan_pass`` from a pytest plugin and reading 57 passing upstream
integration tests as a coverage hole. The patch was in the pytest process; the
code under test was in the kernel.

So the first thing the harness owes anyone is evidence that it landed. These
tests are that evidence, and they are the reason the marker file records a call
COUNT rather than a boolean: "applied" only says the patch was installed, while
"calls" says the mutated code actually ran. A result built on the first without
the second means nothing.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MUTATIONS_DIR = REPO_ROOT / "tests" / "mutations"


def _run_kernelless_probe(tmp_path, mutation: str | None):
    """Run a fresh interpreter that imports cash, and report the marker.

    Deliberately a subprocess rather than an in-process import: the whole point
    is the process boundary, and an in-process check would prove exactly the
    thing that already fooled us.
    """
    marker = tmp_path / "marker.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(MUTATIONS_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    if mutation:
        env["CASH_MUTATION"] = mutation
        env["CASH_MUTATION_MARKER"] = str(marker)
    else:
        env.pop("CASH_MUTATION", None)
        env.pop("CASH_MUTATION_MARKER", None)

    code = (
        "import cash\n"
        # Import the mutation's target, then force one more import so the
        # meta_path applier gets a turn after the target is fully loaded.
        "import cash.notebook.upstream.mismatch_classifier as m\n"
        "import json as _j\n"
        "from cash.notebook.upstream.mismatch_classifier import MismatchClassifier as C\n"
        "print('PATCHED' if C._backward_scan_pass.__name__ == 'dead' else 'ORIGINAL')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True,
        timeout=180, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, f"probe failed: {proc.stderr[-2000:]}"
    found = sorted(tmp_path.glob("marker.*.json"))
    data = json.loads(found[0].read_text(encoding="utf-8")) if found else None
    return proc.stdout.strip(), data


def test_harness_is_inert_without_the_env_var(tmp_path):
    """A stray PYTHONPATH must never change a normal run.

    The harness lives on PYTHONPATH during a mutation run and it would be easy
    to leave it there. If its mere presence altered behaviour, every ordinary
    result afterwards would be suspect.
    """
    stdout, marker = _run_kernelless_probe(tmp_path, mutation=None)
    assert stdout == "ORIGINAL", stdout
    assert marker is None, "wrote a marker with no mutation requested"


def test_mutation_crosses_the_process_boundary(tmp_path):
    """The patch reaches a separate interpreter -- the thing a plugin cannot do."""
    stdout, marker = _run_kernelless_probe(tmp_path, mutation="upstream-dead")
    assert stdout == "PATCHED", stdout
    assert marker is not None, "no marker written"
    assert marker["applied"] is True, marker
    assert marker["pid"] != os.getpid(), "probe ran in THIS process; boundary untested"


def test_an_unknown_mutation_is_loud_and_harmless(tmp_path):
    """A typo must not read as 'the suite tolerated the mutation'."""
    stdout, _ = _run_kernelless_probe(tmp_path, mutation="not-a-real-mutation")
    assert stdout == "ORIGINAL", stdout


@pytest.mark.parametrize("name", ["upstream-dead", "restore-dead", "file-deps-blind"])
def test_every_catalogued_mutation_names_a_real_target(name):
    """Guards against a mutation that can never apply because its target moved.

    Such a mutation is silently inert, and an inert mutation reports the suite
    as tolerant of a breakage that never happened.
    """
    sys.path.insert(0, str(MUTATIONS_DIR))
    try:
        from _catalogue import CATALOGUE
    finally:
        sys.path.pop(0)

    mutation = CATALOGUE[name]
    mod = __import__(mutation.target, fromlist=["_"])
    assert hasattr(mod, mutation.probe), (
        f"{name}: {mutation.target} has no {mutation.probe!r} -- the mutation "
        f"would never apply and would report as tolerated"
    )
    # Every attribute the mutation overwrites must already exist. Patching a
    # name that moved does not fail -- it quietly adds an unused attribute, and
    # the run then reports the suite as tolerant of a breakage that never
    # happened. This exact mistake produced a void `file-deps-blind` result.
    for dotted in mutation.replaces:
        cls_name, _, attr = dotted.rpartition(".")
        owner = getattr(mod, cls_name)
        assert hasattr(owner, attr), (
            f"{name}: {mutation.target}:{dotted} does not exist -- the mutation "
            f"would add a dead attribute and measure nothing"
        )


@pytest.mark.fresh_kernel      # a pooled kernel predates the env change below
def test_the_mutation_actually_runs_inside_a_notebook_kernel(nb_runner, tmp_path):
    """End to end: a real kernel, a real notebook, a non-zero call count.

    This is the assertion the whole harness exists for. Everything above proves
    the patch can cross a process boundary; only this proves it crosses into
    the kernel the integration suite actually drives, and that the broken code
    is on a path a notebook reaches.
    """
    marker = tmp_path / "kernel_marker.json"
    os.environ["CASH_MUTATION"] = "upstream-dead"
    os.environ["CASH_MUTATION_MARKER"] = str(marker)
    old_pp = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(MUTATIONS_DIR) + os.pathsep + old_pp
    try:
        nb_runner.create_notebook([
            "import time\n%load_ext cash\n%cash_on",
            "# @cash:persist\na = 1",
            "# @cash:persist\ntime.sleep(0.05)\nb = a + 1\nprint('b =', b)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        nb_runner.set_cell_source(2, "# @cash:persist\na = 2")
        nb_runner.run_cell(3)          # forces the upstream decision to run
    finally:
        os.environ.pop("CASH_MUTATION", None)
        os.environ.pop("CASH_MUTATION_MARKER", None)
        if old_pp:
            os.environ["PYTHONPATH"] = old_pp
        else:
            os.environ.pop("PYTHONPATH", None)

    found = sorted(tmp_path.glob("kernel_marker.*.json"))
    assert found, (
        "no marker: the mutation never reached the kernel, so any 'suite "
        "tolerated it' conclusion from this harness would be false"
    )
    rows = [json.loads(f.read_text(encoding="utf-8")) for f in found]
    data = max(rows, key=lambda r: r["calls"])
    assert any(r["applied"] for r in rows), rows
    assert data["calls"] > 0, (
        f"applied but never called ({data}) -- the notebook did not reach the "
        f"mutated code, so this run measures nothing"
    )
