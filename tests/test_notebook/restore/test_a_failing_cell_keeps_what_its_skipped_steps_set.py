"""A step the cell plan skipped as overwritten still counts when the step that
overwrites it fails.

Re-running a cell, cash skips a step whose value a later step replaces. When
that later step raises (``data = list(filter(len, ata))``, a typo), the plain
run has already bound ``data`` with the earlier step, so a skipped one must
run before the error leaves the cell (found replaying a student's notebook of
the JuNE dataset, 'student_7' step 83: the list had lost its first element).
"""

import ast

from cash.notebook.ipython.cell_executor import _owed_by_skips

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
