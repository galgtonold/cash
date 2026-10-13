"""A seeded generator handed to a helper of a module of the user's.

The usual Monte Carlo notebook::

    rng = np.random.default_rng(7)          # cell 3
    paths = mc.simulate(5, rng)             # cell 4, slow
    after = rng.standard_normal(2)          # cell 5

cash watched ``rng`` as an argument the helper could change in place, saw
its stream move and refused to store cell 4 ("In-place mutation on: rng"),
so the slow simulation ran on every run, also after a restart that re-ran
the seed. The draw is followed as a draw now: cell 4 is stored with where it
left ``rng``, and a hit puts ``rng`` there, so cell 5 draws what a run
without cash draws. The oracle is the same code run with plain numpy.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]

np = pytest.importorskip("numpy")

HELPER = """import numpy as np, time
def simulate(n, rng):
    time.sleep(0.3)
    z = rng.standard_normal((n, 3))
    return z.sum(axis=1)
"""
# The same paths, but the helper burns EXTRA draws first: the stream below moves.
HELPER_MORE_DRAWS = HELPER.replace("    z = rng", "    rng.standard_normal(EXTRA)\n    z = rng").replace(
    "import numpy as np, time", "import numpy as np, time\nEXTRA = 4"
)

SETUP = "import cash\n%cash_on\n%cash_badge print"
CELLS = [
    SETUP,
    "import numpy as np\nimport mc",
    "rng = np.random.default_rng(7)",
    "paths = mc.simulate(5, rng)\nprint('P', paths.tolist())",
    "after = rng.standard_normal(2)\nprint('A', after.tolist())",
]


def _oracle(seed: int = 7, extra: int = 0) -> tuple[str, str]:
    rng = np.random.default_rng(seed)
    if extra:
        rng.standard_normal(extra)
    paths = rng.standard_normal((5, 3)).sum(axis=1)
    return f"P {paths.tolist()}", f"A {rng.standard_normal(2).tolist()}"


def _assert_matches(nb_runner, seed: int = 7, extra: int = 0) -> None:
    p, a = _oracle(seed, extra)
    assert p in nb_runner.get_output(4), nb_runner.get_output(4)
    assert a in nb_runner.get_output(5), nb_runner.get_output(5)


def _write_helper(nb_runner, source: str) -> None:
    path = nb_runner.work_dir / "mc.py"
    old = path.stat().st_mtime_ns if path.exists() else 0
    path.write_text(source)
    os.utime(path, ns=(old + 2_000_000_000, old + 2_000_000_000))


@pytest.fixture
def notebook(nb_runner):
    _write_helper(nb_runner, HELPER)
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    out = nb_runner.get_output(4)
    assert "EXECUTED: paths = mc.simulate(5, rng)" in out, out
    assert "NOT CACHED" not in out, out
    _assert_matches(nb_runner)
    return nb_runner


def test_the_simulation_is_served_after_a_restart(notebook):
    notebook.restart()
    notebook.run_all()
    out = notebook.get_output(4)
    assert "CACHED: paths = mc.simulate(5, rng)" in out, out
    assert "EXECUTED: paths" not in out, out
    _assert_matches(notebook)


def test_a_draw_below_run_alone_after_a_restart(notebook):
    notebook.restart()
    notebook.run_cell(1)
    notebook.run_cell(5)
    _, a = _oracle()
    assert a in notebook.get_output(5), notebook.get_output(5)


def test_a_reseed_draws_anew_and_the_old_seed_is_served(notebook):
    notebook.set_cell_source(3, "rng = np.random.default_rng(8)")
    notebook.run_cells([3, 4, 5])
    assert "EXECUTED: paths" in notebook.get_output(4)
    _assert_matches(notebook, seed=8)

    notebook.set_cell_source(3, "rng = np.random.default_rng(7)")
    notebook.run_cells([3, 4, 5])
    assert "CACHED: paths" in notebook.get_output(4), notebook.get_output(4)
    _assert_matches(notebook)


def test_a_helper_that_draws_more_moves_the_draws_below(notebook):
    _write_helper(notebook, HELPER_MORE_DRAWS)
    notebook.run_cells([3, 4, 5])
    assert "EXECUTED: paths" in notebook.get_output(4), notebook.get_output(4)
    _assert_matches(notebook, extra=4)

    notebook.restart()
    notebook.run_all()
    assert "CACHED: paths" in notebook.get_output(4), notebook.get_output(4)
    _assert_matches(notebook, extra=4)


def test_the_simulation_run_again_alone_draws_where_a_top_to_bottom_run_does(notebook):
    """Run again without the seed cell, it first rebuilds ``rng`` above it,
    as for a draw written in the cell (see the randomness section of the
    known limitations), and the draw below continues from there."""
    for _ in range(2):
        notebook.run_cell(4)
        out = notebook.get_output(4)
        assert "CACHED: paths" in out and "EXECUTED: paths" not in out, out
        notebook.run_cell(5)
        _assert_matches(notebook)
