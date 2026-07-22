"""CAS-176 verification probe: N cells that write files never reach a stable
state -- and the ticket's stated discriminator (SHARING one state file) is not
the cause.

External witness: the JSON state file(s), read from OUTSIDE the kernel after
each ``run_all``. Each cell bumps its own key, so the counter is an exact
execution count. The badge is not trusted.

The four variants, all with fully disjoint cells (imports hoisted to the setup
cell, per-cell variable names) so no variable edge can explain the result:

  * 1 cell                       -> 1 execution per run_all
  * 4 cells, ONE shared file     -> 4 / 3 / 2 / 1 executions per run_all
  * 4 cells, FOUR separate files -> 4 / 3 / 2 / 1  (byte-identical)

Sharing therefore makes no difference at all. What drives the churn is the
number of *file-writing cells*: every writer cell re-executes every PRECEDING
writer cell's write statement, so a run_all costs N(N+1)/2 executions instead
of N, forever, on an unedited notebook.
"""
import json

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]

RUNS = 3
SETUP = "import cash\n%cash_on\n%cash_badge print\nimport json, time"


def _cell(key, path):
    """Read-modify-write on ``path``, slow enough to clear the 10 ms floor."""
    s = str(path).replace("\\", "/")
    return (
        "# @cash:persist\n"
        f"st_{key} = json.load(open(r'{s}'))\n"
        "time.sleep(0.05)\n"
        f"st_{key}['{key}'] = st_{key}.get('{key}', 0) + 1\n"
        f"with open(r'{s}', 'w') as f_{key}:\n"
        f"    json.dump(st_{key}, f_{key})\n"
        f"v_{key} = st_{key}['{key}']"
    )


def _read(path, key):
    with open(path) as f:
        return json.load(f).get(key, 0)


def _measure(nb_runner, paths, label):
    for p in set(paths.values()):
        p.write_text("{}")
    nb_runner.create_notebook([SETUP] + [_cell(k, p) for k, p in paths.items()])
    nb_runner.start_kernel()

    print(f"\n=== CAS-176 {label} ===")
    history = []
    for n in range(1, RUNS + 1):
        nb_runner.run_all()
        snap = {k: _read(p, k) for k, p in paths.items()}
        history.append(snap)
        print(f"  after run_all #{n}: {snap}")

    # per-run_all execution count = last snapshot / RUNS
    per_run = {k: v / RUNS for k, v in history[-1].items()}
    print(f"  executions per run_all: {per_run}")
    return per_run


def test_single_writer_cell_does_not_amplify(nb_runner, tmp_path):
    """Baseline: one writer cell runs exactly once per run_all."""
    per_run = _measure(
        nb_runner, {"a": tmp_path / "s.json"}, "1 cell (baseline)"
    )
    assert per_run["a"] == 1.0, per_run


def test_four_cells_sharing_one_state_file_amplify_quadratically(nb_runner, tmp_path):
    """The ticket's shape: nothing ever stabilises, and it gets worse per cell."""
    shared = tmp_path / "shared.json"
    per_run = _measure(
        nb_runner, {k: shared for k in ("a", "b", "c", "d")},
        "4 cells, ONE SHARED file",
    )
    assert per_run == {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}, per_run


def test_four_cells_with_separate_files_amplify_identically(nb_runner, tmp_path):
    """The falsifier: swap the shared file for four private ones and NOTHING
    changes. The sharing is not the mechanism."""
    per_run = _measure(
        nb_runner, {k: tmp_path / f"s_{k}.json" for k in ("a", "b", "c", "d")},
        "4 cells, FOUR SEPARATE files",
    )
    assert per_run == {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}, per_run
