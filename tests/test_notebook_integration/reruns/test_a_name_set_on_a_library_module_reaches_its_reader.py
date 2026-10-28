"""A name a statement sets on a library's module moves the module, as one
set on a local module does.

``textwrap.rv = (slow, Z)[1]`` set state on ``textwrap``, but only a local
module's state was looked for. With every statement stored (floors 0), the
reader ``print('R', textwrap.rv)`` was keyed on the import alone: after
``Z = 1`` was edited to ``Z = 2`` it was served, and printed ``R 1`` where
plain Python prints ``R 2``. Set through one name (``import textwrap as
tw; tw.rv = ...``), it moved that name only: a reader binding ``textwrap``
afresh below (``import textwrap``, ``from textwrap import rv``) was served
too. What the notebook set on a library's module is now read by value, as a
local module's data is, by whatever name.

Each shape is checked over Run All x3 (edited after the first), a restart,
and Run All x2.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]

HEAD = (
    "import cash\n%cash_on\nimport time\n"
    "cash.configure(call_cost_floor_seconds=0.0, min_execution_time_to_cache_seconds=0.0)"
)

SHAPES = {
    "attribute": ("import textwrap", "textwrap.rv = (time.sleep(0.3), Z)[1]", "print('R', textwrap.rv)"),
    "setattr": ("import textwrap", "setattr(textwrap, 'rv', (time.sleep(0.3), Z)[1])", "print('R', textwrap.rv)"),
    "another_name": (
        "import textwrap as tw",
        "tw.rv = (time.sleep(0.3), Z)[1]",
        "import textwrap\nprint('R', textwrap.rv)",
    ),
    "taken_by_from": (
        "import textwrap as tw",
        "tw.rv = (time.sleep(0.3), Z)[1]",
        "from textwrap import rv\nprint('R', rv)",
    ),
    "kept_in_a_name": (
        "import textwrap as tw",
        "tw.rv = (time.sleep(0.3), Z)[1]",
        "import textwrap\ny = (time.sleep(0.3), textwrap.rv)[1]\nprint('R', y)",
    ),
    "named_at_run_time": (
        "import textwrap as tw",
        "setattr(tw, 'r' + 'v', (time.sleep(0.3), Z)[1])",
        "import textwrap\nprint('R', textwrap.rv)",
    ),
    "read_by_getattr": (
        "import textwrap as tw",
        "tw.rv = (time.sleep(0.3), Z)[1]",
        "import textwrap\nprint('R', getattr(textwrap, 'rv'))",
    ),
    "set_through_sys_modules": (
        "import sys, textwrap",
        "sys.modules['textwrap'].rv = (time.sleep(0.3), Z)[1]",
        "y = (time.sleep(0.3), textwrap.rv)[1]\nprint('R', y)",
    ),
    "set_through_import_module": (
        "import importlib, textwrap",
        "importlib.import_module('textwrap').rv = (time.sleep(0.3), Z)[1]",
        "y = (time.sleep(0.3), textwrap.rv)[1]\nprint('R', y)",
    ),
    "set_through_a_name_bound_in_the_cell": (
        "import sys, textwrap",
        "m = sys.modules['textwrap']; m.rv = (time.sleep(0.3), Z)[1]",
        "y = (time.sleep(0.3), textwrap.rv)[1]\nprint('R', y)",
    ),
    "set_through_vars": (
        "import textwrap",
        "vars(textwrap)['rv'] = (time.sleep(0.3), Z)[1]",
        "print('R', textwrap.rv)",
    ),
    "read_in_a_function": (
        "import textwrap as tw\nimport textwrap\ndef get():\n    return textwrap.rv",
        "tw.rv = (time.sleep(0.3), Z)[1]",
        "print('R', get())",
    ),
}


@pytest.mark.parametrize("shape", list(SHAPES))
def test_the_reader_prints_what_plain_python_prints(nb_runner, shape):
    imports, writer, reader = SHAPES[shape]
    cells = [HEAD, imports, "Z = 1", writer, reader]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    seen = []
    for run in range(5):
        if run == 1:
            nb_runner.set_cell_source(3, "Z = 2")
        if run == 3:
            nb_runner.restart()
        nb_runner.run_all()
        seen.append([line for line in nb_runner.get_output(len(cells)).splitlines() if line.startswith("R ")])
    assert seen == [["R 1"], ["R 2"], ["R 2"], ["R 2"], ["R 2"]]


def test_a_counter_set_on_a_library_module_draws_as_plain_python_does(nb_runner):
    """``textwrap.cnt = itertools.count()``, then a cheap draw and a slow
    one: each run draws 0 and then 1, as plain Python does."""
    cells = [
        HEAD,
        "import itertools\nimport textwrap",
        "textwrap.cnt = itertools.count()",
        "a = (time.sleep(0.05), next(textwrap.cnt))[1]",
        "b = (time.sleep(0.3), next(textwrap.cnt))[1]\nprint('R', a, b)",
    ]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    seen = []
    for run in range(4):
        if run == 2:
            nb_runner.restart()
        nb_runner.run_all()
        seen.append([line for line in nb_runner.get_output(len(cells)).splitlines() if line.startswith("R ")])
    assert seen == [["R 0 1"]] * 4
