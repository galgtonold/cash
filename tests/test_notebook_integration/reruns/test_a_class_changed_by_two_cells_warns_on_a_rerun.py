"""A class whose state two cells change is kept as it is for a re-run, and
cash says so.

A constructor or method that changes the class itself (``Widget.count += 1``
in ``__init__``) is reset for a cell re-run on its own by re-running the class
def. When another cell changes the class too, that would undo the other
cell's change, so the def is not re-run and the change applies again on top
of the last one. That used to happen silently; it now warns
NOTEBOOK-CLASS-KEPT.
"""

import pytest

pytestmark = [pytest.mark.timeout(300)]

WIDGET = "class Widget:\n    count = 0\n    def __init__(self):\n        Widget.count += 1"


def test_a_rerun_of_one_of_two_cells_changing_the_class_warns(nb_runner):
    nb_runner.create_notebook([WIDGET + "\nw0 = Widget()", "w = Widget()\nprint('count', Widget.count)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "count 2" in nb_runner.get_output(2)
    assert "NOTEBOOK-CLASS-KEPT" not in nb_runner.get_raw_output(2), "a first run starts as plain Jupyter does"
    nb_runner.run_cell(2)
    assert "NOTEBOOK-CLASS-KEPT" in nb_runner.get_raw_output(2)


def test_a_class_only_the_rerun_cell_changes_is_reset_without_a_warning(nb_runner):
    """Control: no other cell changes the class, so its def is re-run."""
    nb_runner.create_notebook([WIDGET, "w = Widget()\nprint('count', Widget.count)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_cell(2)
    assert "count 1" in nb_runner.get_output(2)
    assert "NOTEBOOK-CLASS-KEPT" not in nb_runner.get_raw_output(2)
