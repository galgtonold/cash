"""The check before a cell does no more text work at the end of a long
notebook than near the top.

Before every cell, the upstream check looks at every cell and statement
above it. What their text decides -- a statement's digest, whether it
writes files, what a cell binds -- was worked out again for each of them,
on every cell run: a 400-cell Run All took 49-57 s against 4.4-6.5 s plain.
It is now remembered per text, so the cell at the end of a long notebook
hashes no more than one near the top. What depends on the kernel (values,
files, the environment) is still asked on every check.

Counted in bytes hashed and calls made, not timed.
"""

from __future__ import annotations

from cash.notebook.upstream.read_scope import ReadScope
from tests._cell_driver import run_cash_cell
from tests._work_counts import hashed_bytes


def _cells(n: int, tag: str) -> list[str]:
    cells = [f"{tag}v0 = 1\n{tag}w0 = [0]"]
    cells += [
        f"{tag}v{i} = {tag}v{i - 1} + 1\n{tag}w{i} = {tag}w{i - 1} + [{tag}v{i}]\n{tag}m{i} = len({tag}w{i}) * 2"
        for i in range(1, n)
    ]
    cells.append(f"print({tag}v{n - 1})")
    return cells


def _last_cell(magics, n: int, tag: str, monkeypatch):
    """Bytes hashed by the last cell's run, and how often the files the
    check reads were worked out, in a notebook of *n* chained cells."""
    cells = _cells(n, tag)
    for cell in cells[:-1]:
        run_cash_cell(magics, cell, cells=cells)
    scopes = []
    real = ReadScope.relevant_read_paths

    def counting(self, *args, **kwargs):
        scopes.append(1)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(ReadScope, "relevant_read_paths", counting)
    with hashed_bytes() as hashed:
        run_cash_cell(magics, cells[-1], cells=cells)
    monkeypatch.undo()
    return hashed.bytes, len(scopes)


def test_the_last_cell_hashes_no_more_in_a_long_notebook(cash_magics, monkeypatch):
    short, _ = _last_cell(cash_magics, 10, "short", monkeypatch)
    long, _ = _last_cell(cash_magics, 60, "long", monkeypatch)

    assert short > 0, "nothing was hashed: the counter saw no work"
    # Names are a digit longer down the long notebook; 50 more cells above
    # hashed again would be thousands of bytes.
    assert long <= short + 200, f"{long} bytes hashed at 60 cells against {short} at 10"


def test_no_file_scope_is_worked_out_above_cells_that_write_none(cash_magics, monkeypatch):
    """The files a reconstruction reads matter to a writer only: with none
    above the cell, they are not worked out at all."""
    _, scopes = _last_cell(cash_magics, 30, "scope", monkeypatch)
    assert scopes == 0
