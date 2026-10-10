"""A loop whose iterations were measured heavy keeps one entry per iteration.

The single-unit rule guesses 8 ms of bookkeeping per body statement, which is
right for a tight numeric loop and wrong for one whose iterations each take
longer than the bookkeeping: as one unit, extending ``range(51)`` or editing
the last statement re-runs every iteration. The first run of such a loop is
still one unit, and measures it (``cash/notebook/heavy_loops.py``); runs after
that keep the iterations apart, so an edit re-runs only what it touches.

The seeded tests pin the decision without a clock. The learned one runs a body
of about 0.15 s, well over the 0.1 s the policy needs for three statements;
a slower machine only raises it.
"""

import json

import pytest

pytestmark = [pytest.mark.loops, pytest.mark.timeout(240)]

SINGLE_UNIT = "Fast-loop: executing as single unit"
# ``print`` badge so the CACHED status lands in the cell's text output.
SETUP = "import cash\n%cash_on\n%cash_badge print\n"

# More than 50 iterations and 51 x 3 x 8 ms = 1.2 s of estimated overhead,
# so the static rule sends this loop to one unit.
N = 51
HEAVY = "sum(k % 7 for k in range(3000000))"


def _loop(n=N, tail="res.append(s + i + 1)"):
    return f"res = []\nfor i in range({n}):\n    s = {HEAVY}\n    t = s + i\n    {tail}\ntotal = sum(res)\nprint('total', total)"


def _seed(work_dir, loop_src, seconds):
    import ast

    from cash.notebook.heavy_loops import header_identity

    node = ast.parse(loop_src).body[1]
    cache_dir = work_dir / ".cash"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "_heavy_loops.json").write_text(
        json.dumps({"version": 1, "loops": {header_identity(node): [seconds, ""]}}), encoding="utf-8"
    )


def _expected(n, c):
    base = sum(k % 7 for k in range(3000000))
    return sum(base + i + c for i in range(n))


def test_a_loop_without_a_measurement_runs_as_one_unit(nb_runner):
    nb_runner.create_notebook(["import cash\n%cash_on\n", _loop(tail="res.append(1)")])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert SINGLE_UNIT in nb_runner.get_raw_output(2)


def test_a_loop_measured_heavy_keeps_its_iterations_apart(nb_runner, tmp_path):
    cell = _loop()
    _seed(tmp_path, cell, 1.0)
    nb_runner.create_notebook([SETUP, cell])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert f"total {_expected(N, 1)}" in nb_runner.get_output(2)

    # Edit the last statement: the heavy one is served, the value is right.
    nb_runner.set_cell_source(2, _loop(tail="res.append(s + i + 2)"))
    nb_runner.run_cell(2)
    out = nb_runner.get_output(2)
    assert f"total {_expected(N, 2)}" in out, out
    assert f"LOOP x{N}: s = " in out and f"{N} cached" in out, out


def test_a_measured_cheap_loop_stays_one_unit(nb_runner, tmp_path):
    cell = _loop()
    _seed(tmp_path, cell, 0.0005)
    nb_runner.create_notebook(["import cash\n%cash_on\n", cell])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert SINGLE_UNIT in nb_runner.get_raw_output(2)


def test_a_loop_learns_it_is_heavy_and_then_reuses_an_extension(nb_runner):
    nb_runner.create_notebook([SETUP, _loop()])
    nb_runner.start_kernel()
    nb_runner.run_all()  # one unit: measures
    assert f"total {_expected(N, 1)}" in nb_runner.get_output(2)

    nb_runner.set_cell_source(2, _loop(n=N + 10))
    nb_runner.run_cell(2)  # per iteration: writes the entries
    assert f"total {_expected(N + 10, 1)}" in nb_runner.get_output(2)

    nb_runner.set_cell_source(2, _loop(n=N + 20))
    nb_runner.run_cell(2)  # the first N + 10 iterations come from the cache
    out = nb_runner.get_output(2)
    assert f"total {_expected(N + 20, 1)}" in out, out
    assert f"{N + 10} cached" in out, out


def test_an_unchanged_rerun_of_a_measured_loop_restores_its_unit(nb_runner):
    """The loop ran whole and was measured heavy; run again unchanged it finds
    that unit's entry and restores it, not iteration by iteration."""
    nb_runner.create_notebook([SETUP, _loop()])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_cell(2)
    out = nb_runner.get_output(2)
    assert f"total {_expected(N, 1)}" in out, out
    assert "LOOP x" not in out, out
