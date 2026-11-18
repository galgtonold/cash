"""A process setting a cell's statement makes is in force on every run.

``warnings.simplefilter('error', category=KWarning)`` written straight in
a cell, not in a function: the filters list is the process's, not a
variable's, so a served entry would not put the filter in again, and after
a restart the cells below would run without it. The same for
``warnings.filterwarnings``, ``np.seterr``, a logger's level,
``sys.setrecursionlimit`` and ``atexit.register``. Restart and Run All
finds the setting made. So does a restart that runs only a reader below:
it rebuilt the data the setting's cell made and not the setting, so the
reader ran without the logger's level, the recursion limit or the
``atexit`` callback a Run All gives it.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

IMPORTS = (
    "import atexit, logging, sys, warnings\nimport numpy as np\n"
    "class KWarning(Warning):\n    pass\n"
    "def bye():\n    return None\n"
    "def users_atexit():\n"
    "    try:\n        from cash.tracking.process_registries import _OWN_ATEXIT\n"
    "    except ImportError:\n        return atexit._ncallbacks()\n"
    "    return atexit._ncallbacks() - _OWN_ATEXIT[0]\n"
    "BEFORE = users_atexit()"
)

#: name -> (the cell's statement, an expression true once it is made).
CASES = {
    "simplefilter": (
        "warnings.simplefilter('error', category=KWarning)",
        "any(f[0] == 'error' and f[2] is KWarning for f in warnings.filters)",
    ),
    "filterwarnings": (
        "warnings.filterwarnings('error', message='zz-probe', category=KWarning)",
        "any(f[0] == 'error' and f[2] is KWarning for f in warnings.filters)",
    ),
    "seterr": ("np.seterr(divide='raise')", "np.geterr()['divide'] == 'raise'"),
    "logger-level": ("logging.getLogger('zz.cell').setLevel(5)", "logging.getLogger('zz.cell').level == 5"),
    "recursionlimit": ("sys.setrecursionlimit(4321)", "sys.getrecursionlimit() == 4321"),
    "atexit": ("atexit.register(bye)", "users_atexit() > BEFORE"),
}

READER = "total = sum(range(10))\nprint('MADE', {probe}, total)"


def _cells(kind: str) -> list[str]:
    statement, probe = CASES[kind]
    return ["import cash\n%cash_on", IMPORTS, statement, READER.format(probe=probe)]


@pytest.mark.parametrize("kind", list(CASES))
def test_the_setting_is_made_on_a_run_all_after_a_restart(nb_runner, kind):
    nb_runner.create_notebook(_cells(kind))
    nb_runner.start_kernel()
    for _ in range(2):
        nb_runner.run_all()
        assert "MADE True 45" in nb_runner.get_output(4), nb_runner.get_raw_output(4)

    nb_runner.restart()
    nb_runner._inject_notebook_path()
    nb_runner.run_all()

    assert "MADE True 45" in nb_runner.get_output(4), nb_runner.get_raw_output(4)


@pytest.mark.parametrize("kind", list(CASES))
def test_the_setting_is_made_for_a_reader_run_alone_after_a_restart(nb_runner, kind):
    statement, probe = CASES[kind]
    cells = [
        "import cash\n%cash_on",
        IMPORTS,
        statement + "\ndata = list(range(10))",
        f"total = sum(data)\nprint('MADE', {probe}, total)",
    ]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    for _ in range(2):
        nb_runner.run_all()
    assert "MADE True 45" in nb_runner.get_output(4), nb_runner.get_raw_output(4)

    nb_runner.restart()
    nb_runner._inject_notebook_path()
    nb_runner.run_cell(1)
    nb_runner.run_cell(4)

    assert "MADE True 45" in nb_runner.get_output(4), nb_runner.get_raw_output(4)


def test_a_filter_set_in_a_cell_turns_a_warning_into_an_error_after_a_restart(nb_runner):
    cells = [
        "import cash\n%cash_on",
        "import warnings\nclass KWarning(Warning):\n    pass",
        "warnings.simplefilter('error', category=KWarning)",
        "try:\n    warnings.warn('zz', KWarning)\n    print('PASSED')\nexcept KWarning:\n    print('RAISED')",
    ]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    for _ in range(2):
        nb_runner.run_all()
        assert "RAISED" in nb_runner.get_output(4), nb_runner.get_raw_output(4)

    nb_runner.restart()
    nb_runner._inject_notebook_path()
    nb_runner.run_all()

    assert "RAISED" in nb_runner.get_output(4), nb_runner.get_raw_output(4)
