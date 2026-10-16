"""Cells that each read one file and write it back re-run each other, and are
named once (``NOTEBOOK-FILE-REWRITE-CHAIN``).

Each such cell re-runs the ones above it before it runs, because each write
makes the file the cells above read new again: N of them cost N(N+1)/2 runs
on a Run All. Quiet for one such cell, and for cells that rewrite different
files.
"""

from __future__ import annotations

import warnings

from tests._cell_driver import run_cash_cell

CODE = "NOTEBOOK-FILE-REWRITE-CHAIN"


def _setup(*paths) -> str:
    lines = [f"P{i} = {str(p)!r}\nopen(P{i}, 'w').write('0')" for i, p in enumerate(paths)]
    return "\n".join(lines)


def _step(tag: int, name: str = "P0") -> str:
    return f"tag = {tag}\nv = int(open({name}).read())\nopen({name}, 'w').write(str(v + 1))"


def _run_all(magics, cells: list[str]) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for cell in cells:
            run_cash_cell(magics, cell, cells=cells)
    return [w for w in caught if getattr(w.message, "code", None) == CODE]


def test_three_cells_rewriting_one_file_are_named_once(cash_magics, tmp_path):
    cells = [_setup(tmp_path / "data.csv"), _step(2), _step(3), _step(4)]

    found = _run_all(cash_magics, cells)

    assert len(found) == 1, found
    text = str(found[0].message)
    assert "cells 2 and 3" in text and "data.csv" in text
    assert "re-runs the ones above it" in text
    assert "write its own file" in text
    assert not _run_all(cash_magics, cells), "named again in the same session"


def test_one_cell_rewriting_a_file_is_quiet(cash_magics, tmp_path):
    cells = [_setup(tmp_path / "data.csv"), _step(2), "w = 1"]
    assert not _run_all(cash_magics, cells)
    assert not _run_all(cash_magics, cells)


def test_cells_rewriting_different_files_are_quiet(cash_magics, tmp_path):
    cells = [_setup(tmp_path / "a.csv", tmp_path / "b.csv"), _step(2, "P0"), _step(3, "P1")]
    assert not _run_all(cash_magics, cells)
    assert not _run_all(cash_magics, cells)
