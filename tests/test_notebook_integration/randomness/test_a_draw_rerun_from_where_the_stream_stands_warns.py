"""A draw cash runs again above a cell is redrawn from the seed a run from
the top has, or says it is not (RANDOM-REDRAWN).

Editing ``k`` and running the cell that prints makes cash run ``m`` again.
A literal seed above the draw, and one inside a ``functools.cache`` helper
whose ``def`` cash runs again, are run again with it: the numbers are a run
from the top's, quietly. A seed a module set when it was first imported
cannot run again (a second import does nothing), so cash says so.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(120)]

SEEDS = {
    "in a memo": "@cache\ndef rc(n):\n    random.seed(n)\n    return tuple(range(n))\nsuite = (rc(5), rc(6))",
    "written out": "random.seed(5)\nsuite = (tuple(range(5)), tuple(range(6)))",
}

SEEDED_AT_IMPORT = (
    "import pathlib, sys\n"
    "pathlib.Path('seeds_at_import.py').write_text('import random\\nrandom.seed(6)\\n')\n"
    "sys.path.insert(0, '.')\n"
    "import seeds_at_import\n"
    "suite = (1, 2)"
)


def _cells(seed: str) -> list[str]:
    return [
        "%cash_badge print\nimport random, time\nfrom functools import cache",
        seed,
        "k = 2",
        "m = [k * random.random() for _ in suite] + [time.sleep(0.2)]",
        "print('M', m[:2])",
    ]


@pytest.mark.parametrize("how", list(SEEDS))
def test_a_redrawn_value_is_drawn_from_the_seed_above(nb_runner, how):
    namespace: dict = {}
    top = _cells(SEEDS[how])
    exec("\n".join([top[0].split("\n", 1)[1], top[1], "k = 3", top[3]]).replace("time.sleep(0.2)", "0"), namespace)
    nb_runner.create_notebook(top)
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.set_cell_source(3, "k = 3")
    nb_runner.run_cell(5)
    shown = nb_runner.get_raw_output(5)
    assert f"M {namespace['m'][:2]}" in shown, shown
    assert "RANDOM-REDRAWN" not in shown, shown


def test_a_redrawn_value_whose_seed_ran_at_import_warns(nb_runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    nb_runner.create_notebook(_cells(SEEDED_AT_IMPORT))
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.set_cell_source(3, "k = 3")
    nb_runner.run_cell(5)
    shown = nb_runner.get_raw_output(5)
    assert "RANDOM-REDRAWN" in shown and "m = [k * random.random()" in shown, shown


def test_a_memo_the_plan_defines_again_seeds_again_and_stays_quiet(nb_runner):
    """Edited ``a`` makes cash run ``@cache def rc`` again, so its answers are
    fresh and ``suite = (rc(a), rc(6))`` seeds the stream again before ``m``
    draws: the value is the one a run from the top gives, and no warning."""
    cells = [
        "%cash_badge print\nimport random, time\nfrom functools import cache",
        "@cache\ndef rc(n):\n    random.seed(n)\n    return tuple(range(n))",
        "a = 5",
        "suite = (rc(a), rc(6))",
        "m = [random.random() for _ in suite] + [time.sleep(0.25)]",
        "print('M', m[:2])",
    ]
    namespace: dict = {}
    exec(
        "\n".join([cells[0].split("\n", 1)[1], cells[1], "a = 7", *cells[3:5]]).replace("time.sleep(0.25)", "0"),
        namespace,
    )
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.set_cell_source(3, "a = 7")
    nb_runner.run_cell(6)
    shown = nb_runner.get_raw_output(6)
    assert f"M {namespace['m'][:2]}" in shown, shown
    assert "RANDOM-REDRAWN" not in shown, shown
