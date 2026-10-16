"""A cell using a name imported with ``from m import name`` hits after a restart.

``import math`` then ``x = slow(math.pi)`` restores after a restart; written
``from math import pi`` then ``x = slow(pi)`` it must restore too: on Run All,
and when only the cell is run (its import is run for it), for a constant, a
function, a class, a name from one of the user's modules, and a function of the
notebook that reads such a name as a global.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SLOW = "def slow(v):\n    time.sleep(0.3)\n    return v * 2\n"
HELPERS = "K = 3\ndef f(v):\n    return v + 1\n"
AREA = "def area(r):\n    time.sleep(0.3)\n    return {} * r\n"

CASES = {
    "import_module": ("import math", "x = slow(math.pi)"),
    "from_import": ("from math import pi", "x = slow(pi)"),
    "from_import_as": ("from math import pi as PI", "x = slow(PI)"),
    "function": ("from os.path import join", "x = slow(join('a', 'b'))"),
    "class": ("from collections import OrderedDict", "x = slow(len(OrderedDict(a=1)))"),
    "numpy": ("from numpy import pi as npi", "x = slow(npi)"),
    "users_module": ("from helpers import K, f", "x = slow(f(K))"),
    "global_of_a_function": ("from math import pi", AREA.format("pi") + "x = area(2)"),
    "users_global_of_a_function": ("from helpers import K", AREA.format("K") + "x = area(2)"),
    "star": ("from math import *", "x = slow(pi)"),
}


@pytest.mark.parametrize("rerun", ["run_all", "only_the_cell"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_hits_after_a_restart(nb_runner, case, rerun):
    if case == "star" and rerun == "only_the_cell":
        pytest.skip("a star import binds no name the text shows; plain Python raises NameError here too")
    (nb_runner.work_dir / "helpers.py").write_text(HELPERS, encoding="utf-8")
    imports, use = CASES[case]
    cells = [f"%cash_badge print\nimport time\n{SLOW}", imports, f"{use}\nprint('X', x)"]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = nb_runner.get_raw_output(3)
    assert "X " in first, first
    nb_runner.settle_writes()

    nb_runner.restart()
    nb_runner._init_cash()
    if rerun == "run_all":
        nb_runner.run_all()
    else:
        nb_runner.run_cell(1)
        nb_runner.run_cell(3)
    raw = nb_runner.get_raw_output(3)
    assert "CACHED" in raw, raw
    assert raw.split("X ")[1].split()[0] == first.split("X ")[1].split()[0]
