"""A workbook loaded, formatted and saved every run is not walked for
notebook functions when no cell defined one, and what reads it stays current.

p30-excel-finance's ``format`` cell: ``wb_out = openpyxl.load_workbook(...)``,
two formatting calls, ``wb_out.save(...)``. None of them is ever cached
(each changes the workbook or writes the file it was read from), yet each
walked the workbook's 400,000 cells for the notebook functions it might
hold: five walks a run, 3 s next to 12 s of work. The walk now stops once
it is big and no function with the notebook's globals exists
(`held_functions.notebook_functions_exist`). A function a cell did define
and the workbook holds is still found and still keys its readers.
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest

from cash.notebook import held_functions
from tests._cell_driver import run_cash_cell

MODULE = """
class Cell:
    __slots__ = ("value", "fmt", "parent", "hook")

    def __init__(self, value, parent):
        self.value = value
        self.fmt = None
        self.parent = parent
        self.hook = None


class Book:
    def __init__(self, values):
        self.cells = {(i, 1): Cell(v, self) for i, v in enumerate(values)}


def load(path):
    with open(path) as fh:
        base = float(fh.read().split()[0])
    return Book([base + i for i in range(N)])


def money_format(book):
    for cell in book.cells.values():
        cell.fmt = "#,##0.00"


def attach(book, hook):
    book.cells[(N - 3, 1)].hook = hook


def apply(book):
    return sum(c.hook() for c in book.cells.values() if c.hook is not None)


def save(book, path):
    total = sum(c.value for c in book.cells.values())
    with open(path, "w") as fh:
        fh.write(f"{total}")


def read_total(path):
    with open(path) as fh:
        return float(fh.read())
"""

#: The walk asks after this many objects here (20,000 in use), so the
#: workbook stays small enough for a unit test.
CHECK_AT = 2_000
N = 3 * CHECK_AT


@pytest.fixture
def lib(tmp_path, monkeypatch):
    monkeypatch.setattr(held_functions, "EXISTENCE_CHECK_AT", CHECK_AT)
    # Installed, as openpyxl is: the analysis opens only the objects of the
    # user's own classes to look for code they would run, and a local
    # module's 6,000 cells cost this test 15 s of it under load.
    site = tmp_path / "site-packages"
    site.mkdir()
    name = f"_book_lib_{os.getpid()}_{id(tmp_path)}"
    (site / f"{name}.py").write_text(f"N = {N}\n" + MODULE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(site))
    yield importlib.import_module(name).__name__
    sys.modules.pop(name, None)


@pytest.fixture
def visited(monkeypatch):
    seen = [0]
    real = held_functions._visit_group

    def counting(kind, group, *args):
        seen[0] += len(group)
        return real(kind, group, *args)

    monkeypatch.setattr(held_functions, "_visit_group", counting)
    return seen


def _format_cell(path) -> str:
    return (
        f"wb = lib.load({str(path)!r})\n"
        "lib.money_format(wb)\n"
        f"lib.save(wb, {str(path)!r})\n"
        f"total = lib.read_total({str(path)!r})\n"
    )


def _expected_total(base: float) -> float:
    return float(sum(base + i for i in range(N)))


def test_the_workbook_is_not_walked_and_its_readers_follow_the_file(cash_magics, mock_shell, lib, tmp_path, visited):
    mock_shell.user_ns["__name__"] = "__main__"
    path = tmp_path / "report.txt"
    path.write_text("1.0", encoding="utf-8")
    run_cash_cell(cash_magics, f"import {lib} as lib")
    if held_functions.notebook_functions_exist(mock_shell.user_ns):  # pragma: no cover - a positive control below
        pytest.skip("the test shell itself holds a function with the notebook's globals")
    visited[0] = 0
    run_cash_cell(cash_magics, _format_cell(path))
    run_cash_cell(cash_magics, "report = total * 2")
    assert mock_shell.user_ns["total"] == _expected_total(1.0)
    assert mock_shell.user_ns["report"] == 2 * _expected_total(1.0)
    # Each statement reading wb walked all of its N cells before (2 N here).
    assert visited[0] < N, visited[0]

    # The file is rewritten from outside; the next run reads it again.
    path.write_text("5.0", encoding="utf-8")
    run_cash_cell(cash_magics, _format_cell(path))
    run_cash_cell(cash_magics, "report = total * 2")
    assert mock_shell.user_ns["total"] == _expected_total(5.0)
    assert mock_shell.user_ns["report"] == 2 * _expected_total(5.0)


def test_a_function_a_cell_defined_and_the_workbook_holds_still_keys_its_readers(
    cash_magics, mock_shell, lib, tmp_path, visited
):
    mock_shell.user_ns["__name__"] = "__main__"
    path = tmp_path / "report.txt"
    path.write_text("1.0", encoding="utf-8")
    run_cash_cell(cash_magics, f"import {lib} as lib\nscale = 2")
    run_cash_cell(cash_magics, f"wb = lib.load({str(path)!r})\nlib.attach(wb, lambda: scale * 10)")
    visited[0] = 0
    run_cash_cell(cash_magics, "out = lib.apply(wb)")
    assert mock_shell.user_ns["out"] == 20
    assert visited[0] >= N  # a notebook function exists: walked to the end
    run_cash_cell(cash_magics, "scale = 3")
    run_cash_cell(cash_magics, "out = lib.apply(wb)")
    assert mock_shell.user_ns["out"] == 30


def test_with_a_function_a_cell_defined_the_workbook_is_walked_and_its_readers_follow_the_file(
    cash_magics, mock_shell, lib, tmp_path, visited
):
    # Every real notebook defines one (the replay harness's kernel does): the
    # workbook is then walked to its end, as it may hold the namespace and so
    # every function in it (``test_a_big_value_reaching_a_function_only_
    # through_the_namespace_still_keys_it``), and what reads it stays current.
    mock_shell.user_ns["__name__"] = "__main__"
    path = tmp_path / "report.txt"
    path.write_text("1.0", encoding="utf-8")
    run_cash_cell(cash_magics, f"import {lib} as lib\ndef unrelated():\n    return 1")
    visited[0] = 0
    run_cash_cell(cash_magics, _format_cell(path))
    run_cash_cell(cash_magics, "report = total * 2")
    assert mock_shell.user_ns["total"] == _expected_total(1.0)
    assert mock_shell.user_ns["report"] == 2 * _expected_total(1.0)
    assert visited[0] >= N

    path.write_text("5.0", encoding="utf-8")
    run_cash_cell(cash_magics, _format_cell(path))
    run_cash_cell(cash_magics, "report = total * 2")
    assert mock_shell.user_ns["total"] == _expected_total(5.0)
    assert mock_shell.user_ns["report"] == 2 * _expected_total(5.0)
