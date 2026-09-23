"""The badge's Upstream list is in the order the statements stand in the notebook.

``^CACHED: results[name] = evaluate(...)`` x4 above
``^EXECUTED: results = {}`` -- restores were listed first and re-runs after
them, an order nothing ran in, which made the report hard to read.
"""

SETUP = "import cash\n%load_ext cash\n%cash_badge print\n%cash_on"


def _upstream_rows(out):
    lines = out.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "Upstream:")
    return [line.strip() for line in lines[start + 1 :] if line.strip().startswith("^")]


def _restart_and_run(nb_runner, cell):
    nb_runner.restart()
    nb_runner._inject_notebook_path()
    nb_runner.run_cell(1)
    nb_runner.run_cell(cell)
    return nb_runner.get_output(cell)


def test_a_loops_restored_passes_follow_the_init_that_re_ran(nb_runner):
    nb_runner.create_notebook(
        [
            SETUP,
            "import time\ndef slow(n):\n    time.sleep(0.3)\n    return n * n",
            "results = {}\nfor n in range(2):\n    results[n] = slow(n)\ntotal = sum(results.values())",
            "print('R', sorted(results.items()), total)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    out = _restart_and_run(nb_runner, 4)
    assert "R [(0, 0), (1, 1)] 1" in out, out
    rows = _upstream_rows(out)
    init = next((i for i, r in enumerate(rows) if "results = {}" in r), None)
    passes = [i for i, r in enumerate(rows) if "results[n]" in r or "LOOP" in r]
    total = next((i for i, r in enumerate(rows) if "total = " in r), None)
    assert passes, "\n".join(rows)
    assert init is None or all(init < i for i in passes), "\n".join(rows)
    assert total is None or all(i < total for i in passes), "\n".join(rows)


def test_a_restore_below_a_re_run_is_listed_below_it(nb_runner):
    nb_runner.create_notebook(
        [
            SETUP,
            "import time\nk = 5\nbig = (time.sleep(0.4), k * 1000)[1]",
            "print('B', big, k)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    out = _restart_and_run(nb_runner, 3)
    assert "B 5000 5" in out, out
    rows = _upstream_rows(out)
    print("ROWS", rows)
    k = next((i for i, r in enumerate(rows) if "k = 5" in r), None)
    big = next(i for i, r in enumerate(rows) if "big = " in r)
    assert k is None or k < big, "\n".join(rows)
