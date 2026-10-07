"""After a kernel restart, a cell run alone sees the state the cells above set on a module.

``mylib.configure(5)`` in one cell, ``x = mylib.from_k(2)`` below it: after a
restart, running only the reader imported ``mylib`` again and left the
setting out, so it printed 4 where a plain kernel and Restart & Run All print
10. The same for a setter imported from the module, a notebook helper that
writes into it, and a setting only running it shows (``global K`` in a
method). The runtime now keeps, for a later kernel, which statements set
state on the module, and the rebuild runs them.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

LIB = (
    "K = 2\n"
    "CFG = {'k': 2}\n"
    "def configure(k):\n    global K\n    K = k\n"
    "def set_cfg(k):\n    CFG['k'] = k\n"
    "def from_k(n):\n    return K * n\n"
    "def from_cfg(n):\n    return CFG['k'] * n\n"
    "class Settings:\n    def apply(self, k):\n        global K\n        K = k\n"
)

READER = "x = statelib.from_k(2)\nprint('X', x)"

CASES = {
    "a_module_function_writing_a_global": ["import statelib", "statelib.configure(5)", READER],
    "a_setter_imported_from_the_module": [
        "from statelib import configure, from_k",
        "configure(5)",
        "x = from_k(2)\nprint('X', x)",
    ],
    "a_notebook_helper_writing_into_the_module": [
        "import statelib",
        "def setup(k):\n    global seen\n    seen = k\n    statelib.K = k",
        "setup(5)",
        READER,
    ],
    "a_setter_changing_a_dict_of_the_module": [
        "import statelib",
        "statelib.set_cfg(5)",
        "x = statelib.from_cfg(2)\nprint('X', x)",
    ],
    "a_method_setting_a_global": ["import statelib\ns = statelib.Settings()", "s.apply(5)", READER],
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_reader_run_alone_after_a_restart_sees_the_setting(nb_runner, tmp_path, case):
    (tmp_path / "statelib.py").write_text(LIB, encoding="utf-8")
    cells = CASES[case]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "X 10" in nb_runner.get_output(len(cells))

    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_cell(len(cells))

    assert "X 10" in nb_runner.get_output(len(cells)), nb_runner.get_raw_output(len(cells))
