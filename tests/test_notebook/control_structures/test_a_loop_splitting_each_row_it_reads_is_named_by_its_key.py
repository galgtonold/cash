"""A loop that splits each row it reads is named by its key.

``actions = [x.strip() for x in row[i + 1:].split(sep)]`` for each ``row`` of
a list of 100,000 log lines, in a loop with a ``continue`` (run as one unit): what ``row[i + 1:].split`` is was looked up on
whatever ``row`` was left holding by the loop's last pass, which no lookup
can read, so the loop's outcome was not trusted to be a function of its key.
The list it filled, 1.8M parsed actions, was then read back in full for its
lineage: about 4 s on top of each run of the cell. The method is now read
from the items the loop iterates: a method of a string runs no code of the
user's. A row that is an object of the user's class is still read by its
value, as its ``__getitem__`` and ``split`` are the user's code.
"""

from __future__ import annotations

import ast

import pytest

import cash.notebook.control_structures.helpers as helpers
from cash.analysis.handed_callables import handed_callables
from tests._cell_driver import run_cash_cell

ROWS = "rows = ['@User%d: a (1) -> b (2) -> END (3)' % i for i in range(50)]"
LOOP = (
    "data = []\n"
    "for row in rows:\n"
    "    if not row:\n"
    "        continue\n"
    "    i = row.find(':')\n"
    "    actions = [x.strip() for x in row[i + 1:].split('->')]\n"
    "    data.append([int(row[5:i]), actions])"
)
USERS_ROWS = (
    "class Line:\n"
    "    def __init__(self, text):\n"
    "        self.text = text\n"
    "    def find(self, sep):\n"
    "        return self.text.find(sep)\n"
    "    def __getitem__(self, part):\n"
    "        return Line(self.text[part])\n"
    "    def split(self, sep):\n"
    "        return self.text.split(sep)\n"
    "rows = [Line('@User%d: a (1) -> b (2) -> END (3)' % i) for i in range(50)]"
)
SPLIT = "for row in rows:\n    parts = [x.strip() for x in row[1:].split(',')]"


class Line:
    def __getitem__(self, part):
        return self

    def split(self, sep):
        return [self]

    def strip(self):
        return self


@pytest.fixture
def store_everything(cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0


@pytest.fixture
def read_back(monkeypatch):
    """The names a loop's lineage update read back by their values."""
    names: list[str] = []
    real = helpers.update_mutated_variable_lineages

    def counting(shell, sp, mutated, *args, **kwargs):
        if not kwargs.get("unit_digest"):
            names.extend(sorted(mutated))
        return real(shell, sp, mutated, *args, **kwargs)

    monkeypatch.setattr(helpers, "update_mutated_variable_lineages", counting)
    return names


@pytest.mark.usefixtures("store_everything")
def test_the_filled_list_is_named_by_the_key(cash_magics, mock_shell, read_back):
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    cells = [ROWS, LOOP]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)

    data = mock_shell.user_ns["data"]
    assert data[3] == [3, ["a (1)", "b (2)", "END (3)"]]
    assert len(data) == 50
    assert "data" not in read_back


@pytest.mark.usefixtures("store_everything")
def test_rows_of_the_users_class_are_still_read_by_value(cash_magics, mock_shell, read_back):
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    cells = [USERS_ROWS, LOOP.replace("int(row[5:i])", "int(row.text[5:i])")]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)

    assert mock_shell.user_ns["data"][3] == [3, ["a (1)", "b (2)", "END (3)"]]
    assert "data" in read_back


def test_a_method_of_the_strings_iterated_is_read_from_them():
    tree = ast.parse(SPLIT)
    for namespace in ({"rows": ["a,b", "c"]}, {"rows": ["a,b", "c"], "row": "c"}):
        assert handed_callables(tree, namespace).unreadable == ()


def test_a_method_of_the_users_objects_iterated_cannot_be_told():
    tree = ast.parse(SPLIT)
    for namespace in ({"rows": [Line()]}, {"rows": [Line()], "row": Line()}):
        assert handed_callables(tree, namespace).unreadable
