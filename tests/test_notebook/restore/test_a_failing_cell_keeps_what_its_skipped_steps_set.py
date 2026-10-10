"""A step the cell plan skipped as overwritten still counts when the step that
overwrites it fails.

Re-running a cell, cash skips a step whose value a later step replaces. When
that later step raises (``data = list(filter(len, ata))``, a typo), the plain
run has already bound ``data`` with the earlier step, so a skipped one must
run before the error leaves the cell (found replaying a student's notebook of
the JuNE dataset, 'student_7' step 83: the list had lost its first element).
"""

import ast

import pytest

from cash.notebook.ipython.cell_executor import _owed_by_skips
from tests._cell_driver import run_cash_cell

CELL = """\
PATH = 'f.txt'
data = 'literal'
data = open(PATH).read().split('@')
data = list(filter(len, ata))
n = len(data)
"""


def _owed(skipped, failed, restored=frozenset(), cell=CELL):
    return _owed_by_skips(ast.parse(cell).body, set(skipped), failed, set(restored))


def test_the_last_skipped_writer_before_the_failing_step_is_owed():
    # `data = 'literal'` is overwritten by the open, so only the open is owed.
    # PATH is current: the open reads it as it is.
    assert _owed({0, 1, 2}, failed=3) == ([2], set())


def test_a_step_the_run_wrote_after_is_not_owed():
    # Step 2 ran, so step 1's value is gone in a plain run too.
    assert _owed({0, 1}, failed=3) == ([], set())


def test_nothing_is_owed_when_nothing_was_skipped():
    assert _owed(set(), failed=3) == ([], set())


def test_steps_after_the_failing_one_are_not_owed():
    assert _owed({0, 1, 2, 4}, failed=3) == ([2], set())


def test_a_restored_version_overwritten_after_the_failure_is_owed():
    # The plan restored both versions of data, and the later one, which the
    # cell only reaches after the failing line, is the one the namespace holds.
    cell = "data = a()\ndata = b(data)\nz = boom()\ndata = c(data)\n"
    assert _owed({0}, failed=2, restored={1, 3}, cell=cell) == ([0, 1], set())
    # Restored, with the later version still to run: the namespace holds it.
    assert _owed({0}, failed=2, restored={1}, cell=cell) == ([], set())


def test_a_current_value_no_later_line_writes_is_not_owed():
    # y and w were skipped as current: the namespace already holds them, and
    # running them again would read the x a later line rebound.
    cell = "x = load()\ny = x * 2\nx = heavy(x)\nw = y + 1\nz = boom(x)\nx = x / 2\n"
    assert _owed({1, 3}, failed=4, cell=cell) == ([], set())


def test_an_owed_step_reads_the_version_its_line_saw():
    # y is owed (the line after the failure overwrites it), and it reads the
    # x of line 0, which line 2 rebound: both run again, in order, and x ends
    # as line 2 left it.
    cell = "x = load()\ny = x * 2\nx = heavy(x)\nz = boom(x)\ny = 0\n"
    assert _owed({0, 1}, failed=3, cell=cell) == ([0, 1, 2], set())


def test_a_step_changing_a_value_in_place_is_owed():
    body = "df = make()\ndf['a'] = 1\nbad = nope\ndf = other()\n"
    assert _owed({0, 1}, failed=2, cell=body) == ([0, 1], set())


def test_a_list_the_failed_step_filled_in_place_is_not_owed():
    # Running `rows = []` again after the loop raised would empty what the loop
    # had appended by then (student_7 step 94 of the JuNE dataset).
    body = "rows = []\nfor x in data:\n    rows.append(x.go())\n"
    assert _owed({0}, failed=1, cell=body) == ([], set())


def test_an_owed_step_whose_input_from_before_the_cell_was_rebound_is_lost():
    # y = x * 2 read the x from before the cell, which line 1 rebound: no
    # step brings it back, so y is reported, not recomputed from the new x.
    cell = "y = x * 2\nx = x + 1\nz = boom()\ny = 0\n"
    assert _owed({0}, failed=2, cell=cell) == ([], {"y"})


def test_an_owed_loop_is_lost():
    cell = "y = 0\nfor i in r:\n    y += i\nz = boom()\ny = 1\n"
    assert _owed({0, 1}, failed=2, cell=cell) == ([], {"y"})


def test_a_comprehension_variable_is_not_a_name_of_the_cell():
    cell = "y = [v * 2 for v in x]\nz = boom()\nw = [v for v in y]\n"
    assert _owed({0}, failed=1, cell=cell) == ([], set())


def test_a_failed_loop_leaves_its_lists_unmatched(cash_magics):
    """A re-run of the cell builds its lists again: the failed loop's partial
    appends were recorded against no lineage, so the first run's lists are not
    taken as current (student_7 step 94: 1797 items in a plain kernel, 0 or
    twice as many here)."""
    cell = (
        "PATH = 'f.txt'\ndata = 'literal'\ndata = [1, 2, 3, 'a', 5]\ndata = list(filter(None, data))\n"
        "df = dict(columns=['a'])\nrows, more = [], []\n"
        "for i, x in enumerate(data):\n    rows.append(x + 1)\n    more.append(i)\n"
    )
    cash_magics.cash_on("")
    for _ in range(3):
        with pytest.raises(TypeError):
            run_cash_cell(cash_magics, cell)
        run_cash_cell(cash_magics, "len(rows)")
        assert cash_magics.shell.user_ns["rows"] == [2, 3, 4]
        assert cash_magics.shell.user_ns["more"] == [0, 1, 2]


# -- Re-running a cell that raises again -----------------------------------

HELPERS = """\
import time
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S
def load():
    time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
    return list(range(10))
def heavy(x):
    time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
    return [v + 100 for v in x]
def boom(x):
    raise ValueError("boom")
"""

FAILING = (
    "x = cb6_helpers.load()\n"
    "y = [v * 2 for v in x]\n"
    "x = cb6_helpers.heavy(x)\n"
    "w = [v + 1 for v in y]\n"
    "z = cb6_helpers.boom(x)\n"
    "x = [v / 2 for v in x]"
)


def _run_failing(magics):
    import pytest

    from tests._cell_driver import run_cash_cell

    cells = ["import cb6_helpers", FAILING]
    with pytest.raises(ValueError, match="boom"):
        run_cash_cell(magics, FAILING, cells=cells)


def test_a_second_failing_run_leaves_what_a_plain_run_leaves(cash_magics, mock_shell, tmp_path, monkeypatch):
    # y and w were current: re-running them after the error read the x a later
    # line had already rebound (x = heavy(x)).
    (tmp_path / "cb6_helpers.py").write_text(HELPERS)
    monkeypatch.syspath_prepend(str(tmp_path))
    from tests._cell_driver import run_cash_cell

    run_cash_cell(cash_magics, "import cb6_helpers", cells=["import cb6_helpers", FAILING])
    _run_failing(cash_magics)
    _run_failing(cash_magics)
    ns = mock_shell.user_ns
    assert sum(ns["x"]) == 1045
    assert sum(ns["y"]) == 90
    assert sum(ns["w"]) == 100


def test_an_owed_step_runs_with_the_version_of_its_input_it_saw(cash_magics, mock_shell, tmp_path, monkeypatch):
    # The edited last line raises. The plan restored x = heavy(x) and skipped
    # y = x * 2 (overwritten by the failing line) and x = load(); running y
    # again after the error must not read the x that x = heavy(x) rebound,
    # and x must end as heavy(x) left it.
    (tmp_path / "cb6_helpers.py").write_text(HELPERS)
    monkeypatch.syspath_prepend(str(tmp_path))
    from tests._cell_driver import run_cash_cell

    head = "x = cb6_helpers.load()\ny = [v * 2 for v in x]\nx = cb6_helpers.heavy(x)\n"
    good, bad = head + "y = [v for v in y]", head + "y = cb6_helpers.boom(x)"
    run_cash_cell(cash_magics, "import cb6_helpers", cells=["import cb6_helpers", good])
    run_cash_cell(cash_magics, good, cells=["import cb6_helpers", good])
    del mock_shell.user_ns["x"]  # so the plan restores x = heavy(x) and skips the rest
    with pytest.raises(ValueError, match="boom"):
        run_cash_cell(cash_magics, bad, cells=["import cb6_helpers", bad])
    ns = mock_shell.user_ns
    assert sum(ns["y"]) == 90
    assert sum(ns["x"]) == 1045
