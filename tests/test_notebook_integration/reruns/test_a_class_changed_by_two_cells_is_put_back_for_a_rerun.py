"""A class whose state two cells change is put back for a re-run.

A constructor or method that changes the class itself (``Widget.count += 1``
in ``__init__``) is reset for a cell re-run on its own by re-running the class
def. When another cell changes the class too, that would undo the other
cell's change, so the def is not re-run: the change applied again on top of
the last one, silently. The class's state is now saved where the cell starts
and put back before its re-run, the class staying the same object; when it
cannot be saved, the re-run keeps it and warns NOTEBOOK-CLASS-KEPT.
"""

import pytest

pytestmark = [pytest.mark.timeout(300)]

WIDGET = "class Widget:\n    count = 0\n    def __init__(self):\n        Widget.count += 1"


def _rerun(nb_runner, cells, expect, *, same=()):
    """Run *cells*, then the second again: it prints *expect* both times, and
    each expression in *same* names the same object both times."""
    if same:
        cells = [cells[0], cells[1] + "\nprint('ids', " + ", ".join(f"id({e})" for e in same) + ")"]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = nb_runner.get_output(2)
    assert expect in first, f"first: {first!r}"
    nb_runner.run_cell(2)
    again = nb_runner.get_output(2)
    assert expect in again, f"re-run: {again!r}"
    assert "NOTEBOOK-CLASS-KEPT" not in nb_runner.get_raw_output(2)
    if same:
        ids = [line for line in first.splitlines() if line.startswith("ids")]
        assert ids and ids == [line for line in again.splitlines() if line.startswith("ids")], "same objects"


def test_a_counter_two_cells_bump_is_put_back(nb_runner):
    _rerun(
        nb_runner,
        [WIDGET + "\nw0 = Widget()", "w = Widget()\nprint('count', Widget.count)"],
        "count 2",
        same=["Widget"],
    )


def test_a_registry_on_a_base_is_put_back_on_the_base(nb_runner):
    base = (
        "class Base:\n    registry = []\n    def __init__(self):\n        Base.registry.append(type(self).__name__)\n"
        "class Sub(Base):\n    pass\nb = Base()"
    )
    _rerun(
        nb_runner,
        [base, "s = Sub()\nprint('reg', Base.registry, 'registry' in vars(Sub))"],
        "reg ['Base', 'Sub'] False",
        same=["Base", "Base.registry"],
    )


def test_an_attribute_a_subclass_gains_through_type_self_is_taken_off(nb_runner):
    base = (
        "class Base:\n    count = 0\n    def __init__(self):\n        type(self).count += 1\n"
        "class Sub(Base):\n    pass\nb = Base()"
    )
    _rerun(nb_runner, [base, "s = Sub()\nprint('counts', Base.count, Sub.count)"], "counts 1 2")


def test_a_change_through_self_class_is_put_back(nb_runner):
    widget = (
        "class Widget:\n    count = 0\n    def bump(self):\n        self.__class__.count += 1\nw0 = Widget()\nw0.bump()"
    )
    _rerun(nb_runner, [widget, "w0.bump()\nprint('count', Widget.count)"], "count 2")


def test_a_metaclass_setattr_is_not_run_by_the_put_back(nb_runner):
    meta = (
        "calls = []\n"
        "class Meta(type):\n    def __setattr__(cls, name, value):\n"
        "        calls.append(name)\n        super().__setattr__(name, value)\n"
        "class Widget(metaclass=Meta):\n    count = 0\n    def __init__(self):\n        Widget.count += 1\n"
        "w0 = Widget()"
    )
    # One __setattr__ per run, the constructor's own: the put-back runs none.
    _rerun(nb_runner, [meta, "n = len(calls)\nw = Widget()\nprint('count', Widget.count, len(calls) - n)"], "count 2 1")


def test_state_that_cannot_be_saved_is_kept_with_a_warning(nb_runner):
    widget = (
        "import threading\n"
        "class Widget:\n    count = 0\n    lock = threading.Lock()\n"
        "    def __init__(self):\n        Widget.count += 1\nw0 = Widget()"
    )
    nb_runner.create_notebook([widget, "w = Widget()\nprint('count', Widget.count)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "NOTEBOOK-CLASS-KEPT" not in nb_runner.get_raw_output(2), "a first run starts as plain Jupyter does"
    nb_runner.run_cell(2)
    assert "count 3" in nb_runner.get_output(2)
    assert "NOTEBOOK-CLASS-KEPT" in nb_runner.get_raw_output(2)


def test_a_class_only_the_rerun_cell_changes_is_reset_without_a_warning(nb_runner):
    """Control: no other cell changes the class, so its def is re-run."""
    _rerun(nb_runner, [WIDGET, "w = Widget()\nprint('count', Widget.count)"], "count 1")


def test_a_class_the_rerun_cell_passes_to_a_call_is_put_back(nb_runner):
    """Passing the class on (``id(Widget)``) may call it, so the cell changes
    the class through a second channel; that did re-run the def, while the
    other cell's instance was restored, and lost its change."""
    _rerun(
        nb_runner,
        [WIDGET + "\nw0 = Widget()", "w = Widget()\nprint('count', Widget.count, id(Widget) > 0)"],
        "count 2 True",
    )


@pytest.mark.xfail(
    strict=True,
    reason="already on the C branch: the edited def re-runs, but 'w0 = Widget()', which cell 2 does not read, "
    "is not run again, so its change to Widget.count is missing (11, not 12)",
)
def test_a_class_edited_above_is_rebuilt_not_put_back(nb_runner):
    cells = [WIDGET + "\nw0 = Widget()", "w = Widget()\nprint('count', Widget.count, id(Widget) > 0)"]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(1, cells[0].replace("count = 0", "count = 10"))
    nb_runner.run_cell(2)
    assert "count 12 True" in nb_runner.get_output(2), "as a run from the top gives"
    assert "NOTEBOOK-CLASS-KEPT" not in nb_runner.get_raw_output(2)
