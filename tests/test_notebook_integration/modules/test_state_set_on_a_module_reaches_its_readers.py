"""A cell calling a local module's function follows the state a cell above sets on the module.

``mylib.K = 3`` in one cell, ``b = mylib.from_k(10)`` with ``from_k``
reading ``K`` in the next: editing the setting to 7 and running the notebook
again served 30, with no warning. A plain kernel gives 70.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

LIB = (
    "import time\nK = 2\nCONFIG = {'k': 2}\n"
    "def from_k(x):\n    time.sleep(0.3)\n    return x * K\n"
    "def from_config(x):\n    time.sleep(0.3)\n    return x * CONFIG['k']\n"
)


@pytest.mark.parametrize(
    ("setter", "reader"),
    [
        ("statelib.K = {v}", "b = statelib.from_k(10)"),
        ("statelib.CONFIG['k'] = {v}", "b = statelib.from_config(10)"),
    ],
    ids=["attribute", "item"],
)
def test_editing_the_setting_and_running_all(nb_runner, tmp_path, setter, reader):
    (tmp_path / "statelib.py").write_text(LIB, encoding="utf-8")
    nb_runner.create_notebook(["%cash_badge print\nimport statelib", setter.format(v=3), reader + "\nprint('B', b)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "B 30" in nb_runner.get_output(3)

    nb_runner.set_cell_source(2, setter.format(v=7))
    nb_runner.run_all()
    assert nb_runner.peek("b") == "70", nb_runner.get_raw_output(3)

    # Back to the first value: its entry is still there, and the call is restored.
    nb_runner.set_cell_source(2, setter.format(v=3))
    nb_runner.run_all()
    raw = nb_runner.get_raw_output(3)
    assert nb_runner.peek("b") == "30"
    assert f"CACHED: {reader}" in raw, raw


def test_a_setting_below_the_reader_does_not_reach_it(nb_runner, tmp_path):
    """The data is what the module held when the reader ran. ``mylib.K = 7``
    in a cell below must not send the reader above it to run again with 7."""
    (tmp_path / "statelib.py").write_text(LIB, encoding="utf-8")
    nb_runner.create_notebook(
        ["import statelib", "statelib.K = 3", "b = statelib.from_k(10)", "statelib.K = 7", "u = b + 1\nprint('U', u)"]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "U 31" in nb_runner.get_output(5), nb_runner.get_raw_output(5)

    nb_runner.run_cell(5)
    assert "U 31" in nb_runner.get_output(5), nb_runner.get_raw_output(5)
