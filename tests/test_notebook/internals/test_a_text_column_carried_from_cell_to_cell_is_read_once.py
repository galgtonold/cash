"""An Arrow text column carried from frame to frame, and from cell to cell,
is read for ids once, not once per output per cell.

The identity check reads every Arrow text column of a statement's new frame
for values that spell an ``id()``. Its reading was kept for the cell, by
the Arrow array object: the next cell, or a frame wrapping the same
buffers anew (``chg = labs.merge(...)``'s untouched side), read the column
again: p18 spent 2.5 s reading, 1.1 s now. It is now kept by what names the text (each
chunk's buffers, offset and length) while an array holding those buffers
is alive. That it still finds every id is pinned in
``test_notebook/restore/test_a_text_column_read_in_an_earlier_cell_still_spells_its_ids.py``.
"""

from __future__ import annotations

import pytest

from cash.notebook.statement import identity_holders
from tests._cell_driver import run_cash_cell

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

MAKE = (
    "import pandas as pd\n"
    "df = pd.DataFrame({'s': pd.array([f'w{i}' for i in range(5_000)], dtype='string[pyarrow]'), 'n': range(5_000)})"
)


@pytest.fixture
def reads(monkeypatch) -> list[int]:
    """The length of each Arrow text array read for ids."""
    seen: list[int] = []
    real = identity_holders._spelled_numbers

    def counting(pyarrow, compute, chunked, parts):
        seen.append(len(chunked))
        return real(pyarrow, compute, chunked, parts)

    monkeypatch.setattr(identity_holders, "_spelled_numbers", counting)
    identity_holders.forget_flat_reads()
    yield seen
    identity_holders.forget_flat_reads()


def test_the_next_cells_do_not_read_the_column_again(cash_magics, reads):
    run_cash_cell(cash_magics, MAKE)
    assert reads == [5_000]
    run_cash_cell(cash_magics, "df2 = df.assign(k=1)")
    run_cash_cell(cash_magics, "df3 = df2.rename(columns={'n': 'm'})")
    assert reads == [5_000]


def test_a_new_array_over_the_same_buffers_is_not_read_again(cash_magics, reads):
    run_cash_cell(cash_magics, MAKE)
    run_cash_cell(cash_magics, "m = df.merge(df[['n']], on='n')")
    shell = cash_magics.shell
    old, new = (shell.user_ns[name]["s"].array._pa_array for name in ("df", "m"))
    # Positive control: the merge wrapped the column's buffers anew.
    assert new is not old
    assert new.chunk(0).buffers()[2].address == old.chunk(0).buffers()[2].address
    assert reads == [5_000]


def test_a_changed_column_is_read(cash_magics, reads):
    run_cash_cell(cash_magics, MAKE)
    run_cash_cell(cash_magics, "df['s'] = df['s'] + 'x'")
    assert reads == [5_000, 5_000]


def test_the_reading_goes_with_the_last_array_holding_the_text(reads):
    s = pd.Series([f"w{i}" for i in range(5_000)], dtype="string[pyarrow]")
    assert identity_holders._arrow_text_spelling_numbers(s) is not None
    assert len(identity_holders._ARROW_READS) == 1
    del s
    assert identity_holders._ARROW_READS == {}
    assert identity_holders._ARROW_SIGNED == {}
