"""Several cells that read and write one shared state file.

Mechanism probe: WHY do N writer cells cost N(N+1)/2 executions?

STATUS: mechanism NOT yet identified. This file records what it is NOT, and
the measurement problem that makes it hard to see.

RULED OUT
  * Sharing one state file  — four private files amplify byte-identically
                              (the separate-files variant further down this file).
  * `# @cash:persist`       — present and absent behave the same.
  * In-place vs rebind      — `st[k] = ...` and `st = {**st, ...}` both
                              amplify 4/3/2/1 with an identical file write
                              (test_inplace_vs_rebind_with_identical_file_write).

THE MEASUREMENT PROBLEM — the substantive finding here.

Two witnesses were tried and both are unusable:

1. A counter appended with ``open(path, 'a')``. ``builtins.open`` is precisely
   what cash patches to discover file dependencies, so the counter became a
   dependency of every cell it measured. Every variant amplified, including the
   one designed to contain no file write at all.

2. A raw fd opened once in setup and written with ``os.write`` — genuinely
   untracked (``file_tracker.py`` registers ``builtins.open``, ``io.open``, the
   pandas/polars/numpy/json readers and the directory listers, but not
   ``os.open``/``os.write``). This one does not create a dependency, but adding
   it as a statement in the cell SUPPRESSES the amplification outright: the
   same workload that reads 4/3/2/1 without it reads 1/1/1/1 with it.

So the instrument changes the result in both directions. Every FLAT number in
this file is a fact about the probe, not about the variant. Only a witness that
adds nothing to the cell — the workload's own side effect, as the original
probe uses — measures the real thing.

NEXT STEP: instrument from OUTSIDE the cell instead. Count executions at the
statement-processor level in ``src/cash/notebook/statement/processor.py`` (or
via the upstream reconstruction path in ``upstream/simulator.py``) with a
counter keyed by statement source, and dump it after each ``run_all``. That
observes which statements the reconstruction replays without adding anything to
the user's cell.

Verification probe: N cells that write files never reach a stable
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

That was the finding: every writer cell re-executed every PRECEDING writer
cell's write statement, so a run_all cost N(N+1)/2 executions instead of N,
forever, on an unedited notebook. Since writers are scoped to what the running
cell depends on, four separate files run 1 / 1 / 1 / 1; the shared file, which
each cell really reads, still runs 4 / 3 / 2 / 1.
"""

import json

import pytest

pytestmark = [pytest.mark.integration]


RUNS = 3


KEYS = ("a", "b", "c", "d")


def _setup(counters):
    """Open one append-only fd per cell. os.open/os.write are untracked."""
    lines = ["import cash", "%cash_on", "import json, os, time"]
    for k, c in counters.items():
        p = str(c).replace("\\", "/")
        lines.append(f"_fd_{k} = os.open(r'{p}', os.O_WRONLY | os.O_CREAT | os.O_APPEND)")
    return "\n".join(lines)


def _tick(key):
    return f"os.write(_fd_{key}, b'X')\n"


def _cell_pure(key):
    """CONTROL: expensive, no file I/O whatsoever."""
    return _tick(key) + "time.sleep(0.05)\n" + f"v_{key} = sum(range(200_000))"


def _cell_rmw(key, path):
    """The ticket's shape: read and write the SAME path via builtins.open."""
    p = str(path).replace("\\", "/")
    return (
        _tick(key) + f"st_{key} = json.load(open(r'{p}'))\n"
        "time.sleep(0.05)\n"
        f"st_{key}['{key}'] = st_{key}.get('{key}', 0) + 1\n"
        f"with open(r'{p}', 'w') as f_{key}:\n"
        f"    json.dump(st_{key}, f_{key})\n"
        f"v_{key} = st_{key}['{key}']"
    )


def _cell_read_only(key, path):
    """Reads a file, never writes one."""
    p = str(path).replace("\\", "/")
    return (
        _tick(key) + f"st_{key} = json.load(open(r'{p}'))\n"
        "time.sleep(0.05)\n"
        f"v_{key} = st_{key}.get('seed', 0) + {len(key)}"
    )


def _cell_write_only(key, dst):
    """Writes a file, never reads one."""
    d = str(dst).replace("\\", "/")
    return (
        _tick(key) + "time.sleep(0.05)\n"
        f"st_{key} = {{'{key}': 1}}\n"
        f"with open(r'{d}', 'w') as f_{key}:\n"
        f"    json.dump(st_{key}, f_{key})\n"
        f"v_{key} = st_{key}['{key}']"
    )


def _count(counter):
    try:
        return len(counter.read_bytes())
    except FileNotFoundError:
        return 0


def _measure_executions(nb_runner, cells, counters, label):
    nb_runner.create_notebook([_setup(counters)] + cells)
    nb_runner.start_kernel()

    print(f"\n=== Shared-state-file mechanism: {label} ===")
    for n in range(1, RUNS + 1):
        nb_runner.run_all()
        print(f"  after run_all #{n}: { {k: _count(c) for k, c in counters.items()} }")

    per_run = {k: _count(c) / RUNS for k, c in counters.items()}
    print(f"  executions per run_all: {per_run}")
    return per_run


@pytest.fixture
def counters(tmp_path):
    return {k: tmp_path / f"count_{k}.log" for k in KEYS}


FLAT = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}


QUADRATIC = {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}


@pytest.mark.timeout(900)
def test_pure_compute_does_not_amplify(nb_runner, counters):
    """CONTROL. Also validates the witness: if os.write were tracked, this
    would amplify and every other result in this file would be meaningless."""
    per_run = _measure_executions(
        nb_runner,
        [_cell_pure(k) for k in KEYS],
        counters,
        "pure compute, no file I/O",
    )
    assert per_run == FLAT, per_run


@pytest.mark.timeout(900)
def test_read_write_same_path_amplifies(nb_runner, tmp_path, counters):
    """DOES NOT amplify — and that result is about the WITNESS, not the shape.

    The identical workload amplifies 4/3/2/1 in the original probe and in
    test_inplace_vs_rebind_with_identical_file_write below. The only difference
    here is the `os.write(_fd_k, ...)` tick added as the cell's FIRST
    statement. Adding it flattens the cell completely.

    So an in-cell counter does not merely fail to observe the amplification, it
    PREVENTS it: the extra statement changes cash's caching decision for the
    cell. Every "flat" result in this file is therefore evidence about the
    instrument, not about the variant it was meant to test.
    """
    state = {k: tmp_path / f"s_{k}.json" for k in KEYS}
    for p in state.values():
        p.write_text("{}")
    per_run = _measure_executions(
        nb_runner,
        [_cell_rmw(k, state[k]) for k in KEYS],
        counters,
        "read X / write X (ticket shape)",
    )
    assert per_run == FLAT, per_run  # suppressed by the witness — see docstring


@pytest.mark.timeout(900)
def test_read_only_does_not_amplify(nb_runner, tmp_path, counters):
    """Does merely DEPENDING on a file amplify, with nothing written?"""
    src = {k: tmp_path / f"r_{k}.json" for k in KEYS}
    for p in src.values():
        p.write_text('{"seed": 1}')
    per_run = _measure_executions(
        nb_runner,
        [_cell_read_only(k, src[k]) for k in KEYS],
        counters,
        "read only, no write",
    )
    assert per_run == FLAT, per_run


@pytest.mark.timeout(900)
def test_write_only(nb_runner, tmp_path, counters):
    """Does WRITING alone amplify, with no file dependency to invalidate?"""
    dst = {k: tmp_path / f"w_{k}.json" for k in KEYS}
    per_run = _measure_executions(
        nb_runner,
        [_cell_write_only(k, dst[k]) for k in KEYS],
        counters,
        "write only, no read",
    )
    assert per_run == FLAT, per_run


@pytest.mark.timeout(900)
def test_persist_on_a_file_writer_amplifies(nb_runner, tmp_path, counters):
    """Persist is NOT the trigger: this is flat, like its unannotated twin.

    Both arms carry the in-cell witness, so both are suppressed (see
    test_read_write_same_path_amplifies). What this rules out is persist as an
    ADDITIONAL trigger on top of the witness.
    """
    state = {k: tmp_path / f"p_{k}.json" for k in KEYS}
    for p in state.values():
        p.write_text("{}")
    per_run = _measure_executions(
        nb_runner,
        ["# @cash:persist\n" + _cell_rmw(k, state[k]) for k in KEYS],
        counters,
        "read X / write X WITH @cash:persist",
    )
    assert per_run == FLAT, per_run


@pytest.mark.timeout(900)
def test_persist_on_pure_compute(nb_runner, counters):
    """Isolates persist from files entirely: does persist alone amplify?"""
    per_run = _measure_executions(
        nb_runner,
        ["# @cash:persist\n" + _cell_pure(k) for k in KEYS],
        counters,
        "pure compute WITH @cash:persist",
    )
    assert per_run == FLAT, per_run


@pytest.mark.timeout(900)
def test_per_statement_execution_profile(nb_runner, tmp_path):
    """Locate the amplification by statement position within one cell.

    Not an assertion of expected values — a measurement. It prints, per cell,
    how many times each statement position executed per run_all.
    """
    POSITIONS = ("load", "sleep", "incr", "write", "tail")
    fds = {(k, pos): tmp_path / f"c_{k}_{pos}.log" for k in KEYS for pos in POSITIONS}

    setup_lines = ["import cash", "%cash_on", "import json, os, time"]
    for (k, pos), p in fds.items():
        s = str(p).replace("\\", "/")
        setup_lines.append(f"_fd_{k}_{pos} = os.open(r'{s}', os.O_WRONLY|os.O_CREAT|os.O_APPEND)")

    state = {k: tmp_path / f"st_{k}.json" for k in KEYS}
    for p in state.values():
        p.write_text("{}")

    def cell(k):
        s = str(state[k]).replace("\\", "/")
        return (
            "# @cash:persist\n"
            f"st_{k} = json.load(open(r'{s}'))\n"
            f"os.write(_fd_{k}_load, b'X')\n"
            "time.sleep(0.05)\n"
            f"os.write(_fd_{k}_sleep, b'X')\n"
            f"st_{k}['{k}'] = st_{k}.get('{k}', 0) + 1\n"
            f"os.write(_fd_{k}_incr, b'X')\n"
            f"with open(r'{s}', 'w') as f_{k}:\n"
            f"    json.dump(st_{k}, f_{k})\n"
            f"os.write(_fd_{k}_write, b'X')\n"
            f"v_{k} = st_{k}['{k}']\n"
            f"os.write(_fd_{k}_tail, b'X')"
        )

    nb_runner.create_notebook(["\n".join(setup_lines)] + [cell(k) for k in KEYS])
    nb_runner.start_kernel()
    for _ in range(RUNS):
        nb_runner.run_all()

    print("\n=== Per-statement execution profile (per run_all) ===")
    print(f"  {'cell':<6}" + "".join(f"{p:>9}" for p in POSITIONS))
    for k in KEYS:
        row = "".join(f"{_count(fds[(k, pos)]) / RUNS:>9.2f}" for pos in POSITIONS)
        print(f"  {k:<6}{row}")
    print(f"  state file totals: { {k: __import__('json').load(open(p)).get(k, 0) / RUNS for k, p in state.items()} }")


@pytest.mark.timeout(900)
def test_fused_witness_execution_profile(nb_runner, tmp_path):
    """The previous probe reported every statement at 1.00 while the state file
    read 4/3/2/1 — impossible if the increment really ran once.

    The resolution is that a witness on its OWN line is a separate statement.
    Cash re-executes individual statements, so replaying just the increment or
    just the write never touches the tick beside it. Here the witness is fused
    INTO the statement it measures, via helpers defined in the setup cell, so a
    replay of that statement necessarily fires its counter.
    """
    parts = ("incr", "write")
    fds = {(k, p): tmp_path / f"f_{k}_{p}.log" for k in KEYS for p in parts}

    setup = ["import cash", "%cash_on", "import json, os, time"]
    for (k, p), path in fds.items():
        s = str(path).replace("\\", "/")
        setup.append(f"_fd_{k}_{p} = os.open(r'{s}', os.O_WRONLY|os.O_CREAT|os.O_APPEND)")
    setup += [
        "def _bump(d, key, fd):",
        "    os.write(fd, b'X')",
        "    d[key] = d.get(key, 0) + 1",
        "    return d",
        "def _dump(d, path, fd):",
        "    os.write(fd, b'X')",
        "    with open(path, 'w') as f:",
        "        json.dump(d, f)",
    ]

    state = {k: tmp_path / f"fs_{k}.json" for k in KEYS}
    for p in state.values():
        p.write_text("{}")

    def cell(k):
        s = str(state[k]).replace("\\", "/")
        return (
            "# @cash:persist\n"
            f"st_{k} = json.load(open(r'{s}'))\n"
            "time.sleep(0.05)\n"
            f"st_{k} = _bump(st_{k}, '{k}', _fd_{k}_incr)\n"
            f"_dump(st_{k}, r'{s}', _fd_{k}_write)\n"
            f"v_{k} = st_{k}['{k}']"
        )

    nb_runner.create_notebook(["\n".join(setup)] + [cell(k) for k in KEYS])
    nb_runner.start_kernel()
    for _ in range(RUNS):
        nb_runner.run_all()

    print("\n=== FUSED-witness profile (executions per run_all) ===")
    print(f"  {'cell':<6}{'incr':>9}{'write':>9}{'file':>9}")
    import json as _json

    for k in KEYS:
        incr = _count(fds[(k, "incr")]) / RUNS
        write = _count(fds[(k, "write")]) / RUNS
        val = _json.load(open(state[k])).get(k, 0) / RUNS
        print(f"  {k:<6}{incr:>9.2f}{write:>9.2f}{val:>9.2f}")


@pytest.mark.timeout(900)
def test_inplace_vs_rebind_with_identical_file_write(nb_runner, tmp_path):
    """The last variable. Both arms below do the SAME inline file write; they
    differ only in whether the dict is mutated in place or rebound.

    If only the in-place arm amplifies, the amplification is not a file-writing defect at
    all — it is the in-place-mutation rebuild rule being
    charged once per downstream cell.
    """
    import json as _json

    results = {}
    for label, incr in (
        ("in-place  st[k] = ...", "st_{k}['{k}'] = st_{k}.get('{k}', 0) + 1"),
        ("rebind    st = {{**st}}", "st_{k} = {{**st_{k}, '{k}': st_{k}.get('{k}', 0) + 1}}"),
    ):
        state = {k: tmp_path / f"{label[:7].strip()}_{k}.json" for k in KEYS}
        for p in state.values():
            p.write_text("{}")

        def cell(k):
            s = str(state[k]).replace("\\", "/")
            return (
                "# @cash:persist\n"
                f"st_{k} = json.load(open(r'{s}'))\n"
                "time.sleep(0.05)\n" + incr.format(k=k) + "\n"
                f"with open(r'{s}', 'w') as f_{k}:\n"
                f"    json.dump(st_{k}, f_{k})\n"
                f"v_{k} = st_{k}['{k}']"
            )

        nb_runner.create_notebook(["import cash\n%cash_on\nimport json, os, time"] + [cell(k) for k in KEYS])
        nb_runner.start_kernel()
        for _ in range(RUNS):
            nb_runner.run_all()
        results[label] = {k: _json.load(open(p)).get(k, 0) / RUNS for k, p in state.items()}

    print("\n=== In-place vs rebind (identical file write) ===")
    for label, per_run in results.items():
        print(f"  {label:<26} {per_run}")


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

    print(f"\n=== shared state file {label} ===")
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


@pytest.mark.timeout(600)
def test_single_writer_cell_does_not_amplify(nb_runner, tmp_path):
    """Baseline: one writer cell runs exactly once per run_all."""
    per_run = _measure(nb_runner, {"a": tmp_path / "s.json"}, "1 cell (baseline)")
    assert per_run["a"] == 1.0, per_run


@pytest.mark.timeout(600)
def test_four_cells_sharing_one_state_file_amplify_quadratically(nb_runner, tmp_path):
    """The ticket's shape: nothing ever stabilises, and it gets worse per cell."""
    shared = tmp_path / "shared.json"
    per_run = _measure(
        nb_runner,
        {k: shared for k in ("a", "b", "c", "d")},
        "4 cells, ONE SHARED file",
    )
    assert per_run == {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}, per_run


@pytest.mark.timeout(600)
def test_four_cells_with_separate_files_run_once_each(nb_runner, tmp_path):
    """The falsifier used to show four private files amplifying exactly like one
    shared file. A writer is now re-run only for a cell that depends on what it
    wrote, so with private files every cell runs once per run_all, as in a
    plain kernel. The shared file still amplifies: each cell
    reads the file the one before it wrote."""
    per_run = _measure(
        nb_runner,
        {k: tmp_path / f"s_{k}.json" for k in ("a", "b", "c", "d")},
        "4 cells, FOUR SEPARATE files",
    )
    assert per_run == {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}, per_run
