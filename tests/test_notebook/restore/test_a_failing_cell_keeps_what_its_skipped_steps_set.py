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


def _owed(skipped, failed):
    return _owed_by_skips(ast.parse(CELL).body, set(skipped), failed)


def test_the_last_skipped_writer_before_the_failing_step_is_owed():
    # `data = 'literal'` is overwritten by the open, so only the open is owed,
    # and it reads PATH, which is owed as well.
    assert _owed({0, 1, 2}, failed=3) == [0, 2]


def test_a_step_the_run_wrote_after_is_not_owed():
    # Step 2 ran, so step 1's value is gone in a plain run too.
    assert _owed({0, 1}, failed=3) == [0]


def test_nothing_is_owed_when_nothing_was_skipped():
    assert _owed(set(), failed=3) == []


def test_steps_after_the_failing_one_are_not_owed():
    assert _owed({0, 1, 2, 4}, failed=3) == [0, 2]


def test_a_step_changing_a_value_in_place_is_owed():
    body = ast.parse("df = make()\ndf['a'] = 1\nbad = nope\n").body
    assert _owed_by_skips(body, {0, 1}, 2) == [0, 1]


def test_a_list_the_failed_step_filled_in_place_is_not_owed():
    # Running `rows = []` again after the loop raised would empty what the loop
    # had appended by then (student_7 step 94 of the JuNE dataset).
    body = ast.parse("rows = []\nfor x in data:\n    rows.append(x.go())\n").body
    assert _owed_by_skips(body, {0}, 1) == []


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
