"""A call through what holds a notebook function sees a global the function
reads edited, in a real kernel, with or without the notebook file.

``fns = [make]`` and ``r = fns[0](3)``, with ``make`` reading ``cfg``: run
headless (``NOTEBOOK-NOT-FOUND``), cash served the old ``r`` after ``cfg``
was edited and run, as it did for a dict of builders, a partial, a lambda in
a dict, a function kept as a default, a method of a notebook class and a
``FunctionTransformer``. With the notebook file the upstream check rebuilt
it. The key now reads what the held function reads, at the call.
"""

import pytest

from tests._nbharness.badge import shows_cached

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300), pytest.mark.fresh_kernel]

SETUP = "import cash\n%load_ext cash\n%cash_badge print\n%cash_on"
MAKE = "def make(n):\n    return [cfg['w'] * n]"

#: id -> (the definitions, what holds the function, the call through it)
SHAPES = {
    "list": (MAKE, "fns = [make]", "fns[0](3)"),
    "dict": (MAKE, "builders = {'a': make, 'b': make}", "builders['a'](3)"),
    "partial": ("import functools\n" + MAKE, "p = functools.partial(make, 3)", "p()"),
    "lambda": ("pass", "d = {'f': lambda n: [cfg['w'] * n]}", "d['f'](3)"),
    "default": (MAKE, "def run(f=make):\n    return f(3)", "run()"),
    "method": ("class M:\n    def build(self, n):\n        return [cfg['w'] * n]", "m = M()", "m.build(3)"),
    "transformer": (
        "import numpy as np\nfrom sklearn.preprocessing import FunctionTransformer\n"
        "def make(X):\n    return (np.asarray(X) * cfg['w']).ravel().tolist()",
        "ft = FunctionTransformer(make)",
        "ft.transform([[3]])",
    ),
}

CASES = [pytest.param(shape, False, id=f"{shape}-headless") for shape in SHAPES] + [
    pytest.param(shape, True, id=f"{shape}-notebook") for shape in ("list", "method")
]


def _result(nb_runner):
    out = nb_runner.get_output(5)
    return out, [line for line in out.splitlines() if line.startswith("R ")]


@pytest.mark.parametrize(("shape", "notebook_known"), CASES)
def test_the_call_sees_the_edited_global(nb_runner, shape, notebook_known):
    definitions, hold, call = SHAPES[shape]
    nb_runner.create_notebook(
        [SETUP, "import time\ncfg = {'w': 1}", definitions, hold, f"r = (time.sleep(0.6), {call})[1]\nprint('R', r)"]
    )
    nb_runner.start_kernel(inject_notebook_path=notebook_known)
    nb_runner.run_all()
    out, printed = _result(nb_runner)
    assert printed == ["R [3]"], out
    nb_runner.run_cells([5])
    out, printed = _result(nb_runner)
    assert printed == ["R [3]"] and shows_cached(out), f"control: unchanged, served\n{out}"

    nb_runner.set_cell_source(2, "import time\ncfg = {'w': 2}")
    nb_runner.run_cells([2, 5])
    out, printed = _result(nb_runner)
    assert printed == ["R [6]"], out

    nb_runner.run_cells([5])
    out, printed = _result(nb_runner)
    assert printed == ["R [6]"] and shows_cached(out), f"the new result is stored\n{out}"

    nb_runner.restart()
    nb_runner.run_all()
    out, printed = _result(nb_runner)
    assert printed == ["R [6]"], out
