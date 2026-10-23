"""A draw cash runs again above a cell is drawn from the seed a run from the
top has, also when the seed is one cash only sees at run time: a library
function that seeds (``rsclib.setup(4)``), a seed in ``__init__``
(``Exp(3)``), or a temp seed a ``with`` sets and puts back. A temp seed
seeds nothing below its ``with``, so a draw there follows the seed above
it. The numbers are a run from the top's, and nothing warns.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(120)]

LIBRARY = "import random\ndef setup(s):\n    random.seed(s)\n"
TEMP_SEED = (
    "@contextlib.contextmanager\ndef temp_seed(s):\n    st = random.getstate()\n    random.seed(s)\n"
    "    try:\n        yield\n    finally:\n        random.setstate(st)"
)
DRAW = "m = [k * random.random() for _ in range(2)] + [time.sleep(0.2)]"

CASES = {
    "library function": (
        ["import random, time, sys\nsys.path.insert(0, {lib!r})\nimport rsclib", "rsclib.setup(4)", "k = 2", DRAW],
        3,
        "k = 3",
    ),
    "seed in __init__": (
        [
            "import random, time",
            "class Exp:\n    def __init__(self, s):\n        random.seed(s)",
            "e = Exp(3)",
            "k = 2",
            DRAW,
        ],
        4,
        "k = 3",
    ),
    "temp seed": (
        ["import random, time, contextlib", TEMP_SEED, "s = 123", "k = 1\nwith temp_seed(s):\n    " + DRAW],
        3,
        "s = 7",
    ),
    "below a temp seed": (
        [
            "import random, time, contextlib",
            TEMP_SEED,
            "random.seed(1)",
            "with temp_seed(123):\n    a = 1",
            "k = 1",
            DRAW,
        ],
        3,
        "random.seed(2)",
    ),
}


@pytest.mark.parametrize("how", list(CASES))
def test_a_redrawn_value_is_drawn_from_the_seed_above(nb_runner, how, tmp_path):
    (tmp_path / "rsclib.py").write_text(LIBRARY)
    cells, edited, edit = CASES[how]
    cells = [cell.format(lib=str(tmp_path)) for cell in cells]
    plain = list(cells)
    plain[edited - 1] = edit
    namespace: dict = {}
    exec("\n".join(plain).replace("time.sleep(0.2)", "0"), namespace)

    nb_runner.create_notebook(["%cash_badge print\n" + cells[0], *cells[1:], "print('M', m[:2])"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(edited, edit)
    last = len(cells) + 1
    nb_runner.run_cell(last)
    shown = nb_runner.get_raw_output(last)
    assert f"M {namespace['m'][:2]}" in shown, shown
    assert "RANDOM-REDRAWN" not in shown, shown


def test_after_a_restart_a_temp_seed_helper_runs_after_the_import_it_reads(nb_runner):
    """The import shares its cell with a magic line; the chain cash runs again
    after a restart still runs it ahead of the ``def`` that reads it."""
    cells, _edited, _edit = CASES["below a temp seed"]
    namespace: dict = {}
    exec("\n".join(cells).replace("time.sleep(0.2)", "0"), namespace)

    nb_runner.create_notebook(["%cash_badge print\n" + cells[0], *cells[1:], "print('M', m[:2])"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.restart()
    nb_runner._init_cash()
    last = len(cells) + 1
    nb_runner.run_cell(last)
    shown = nb_runner.get_raw_output(last)
    assert f"M {namespace['m'][:2]}" in shown, shown


@pytest.mark.parametrize("how", ["library function", "seed in __init__"])
def test_after_a_restart_the_last_cell_alone_draws_from_the_seed(nb_runner, how, tmp_path):
    """The seed cash saw when the statement ran is kept across the restart,
    so the draw it rebuilds for the last cell is a run from the top's."""
    (tmp_path / "rsclib.py").write_text(LIBRARY)
    cells, _edited, _edit = CASES[how]
    cells = [cell.format(lib=str(tmp_path)) for cell in cells]
    namespace: dict = {}
    exec("\n".join(cells).replace("time.sleep(0.2)", "0"), namespace)

    nb_runner.create_notebook(["%cash_badge print\n" + cells[0], *cells[1:], "print('M', m[:2])"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.restart()
    nb_runner._init_cash()
    last = len(cells) + 1
    nb_runner.run_cell(last)
    shown = nb_runner.get_raw_output(last)
    assert f"M {namespace['m'][:2]}" in shown, shown


def test_a_draw_below_an_import_that_seeds_is_drawn_once_on_the_first_run_all(nb_runner, tmp_path):
    """``import seedmod`` seeds ``random`` as it loads. The first Run All
    draws ``x`` once: the cell that prints it does not draw it again."""
    (tmp_path / "seedmod.py").write_text("import random\nrandom.seed(11)\n")
    cells = [
        f"%cash_badge print\nimport random, sys\nsys.path.insert(0, {str(tmp_path)!r})\nimport seedmod",
        "x = random.random()",
        "print('X', x)",
    ]
    import random

    first = random.Random(11).random()
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    shown = nb_runner.get_raw_output(3)
    assert f"X {first}" in shown and "Upstream" not in shown, shown
