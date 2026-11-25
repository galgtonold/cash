"""A name a ``match`` pattern captures is a variable the statement writes.

``case {'k': [a, *rest]}``, ``case Point(x=px)``, ``case [first, *others]``,
``case x if ...`` and ``case ... as whole`` bind names with no assignment in
sight. Read as written by the ``match`` statement, a cell below that uses
them is brought up to date like after any other assignment: after an edit to
what is matched, and after a kernel restart, it gets what a run from the top
gives -- not the value from before the edit, the value of an earlier
assignment to the same name, or a NameError.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = "from dataclasses import dataclass\n@dataclass\nclass Point:\n    x: int\n    y: int"

# name: (cell defining `data`, the same cell edited, the match statement, the reader, printed before, printed after)
SHAPES = {
    "a_mapping_holding_a_sequence": (
        "a = 0\ndata = {'k': [1, 2, 3]}",
        "a = 0\ndata = {'k': [7, 8]}",
        "match data:\n    case {'k': [a, *rest]}:\n        pass",
        "print('R', a, rest)",
        "R 1 [2, 3]",
        "R 7 [8]",
    ),
    "the_rest_of_a_mapping": (
        "data = {'k': 1, 'm': 2}",
        "data = {'k': 5, 'n': 6}",
        "match data:\n    case {'k': v, **others}:\n        pass",
        "print('R', v, others)",
        "R 1 {'m': 2}",
        "R 5 {'n': 6}",
    ),
    "a_class_pattern": (
        "px = -1\ndata = Point(1, 2)",
        "px = -1\ndata = Point(9, 2)",
        "match data:\n    case Point(x=px, y=2):\n        pass",
        "print('R', px)",
        "R 1",
        "R 9",
    ),
    "a_capture_with_a_guard": (
        "data = 3",
        "data = 4",
        "match data:\n    case x if x > 0:\n        pass",
        "print('R', x)",
        "R 3",
        "R 4",
    ),
    "an_as_capture": (
        "data = [1, 2]",
        "data = [5, 6]",
        "match data:\n    case [_, _] as whole:\n        pass",
        "print('R', whole)",
        "R [1, 2]",
        "R [5, 6]",
    ),
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_cell_below_follows_an_edit_to_what_is_matched(nb_runner, shape):
    before, edited, match, reader, first, second = SHAPES[shape]
    nb_runner.create_notebook([SETUP, before, match, reader])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert first in nb_runner.get_output(4), "control: the top-to-bottom run"

    nb_runner.set_cell_source(2, edited)
    nb_runner.run_cell(4)

    assert second in nb_runner.get_output(4), nb_runner.get_raw_output(4)


@pytest.mark.fresh_kernel
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_cell_below_run_alone_after_a_restart_gets_the_captured_values(nb_runner, shape):
    before, _edited, match, reader, first, _second = SHAPES[shape]
    nb_runner.create_notebook([SETUP, before, match, reader])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert first in nb_runner.get_output(4), "control: the top-to-bottom run"

    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_cell(4)

    assert first in nb_runner.get_output(4), nb_runner.get_raw_output(4)
