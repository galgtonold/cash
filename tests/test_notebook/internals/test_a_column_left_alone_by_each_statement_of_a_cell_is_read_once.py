"""A frame's big column of sessions, left as it was by each statement of a
cell that adds a column, is read for ids once, not after every statement,
and not again by the next cell.

Each ``df['k'] = ...`` changes ``df`` in place, so its value is checked for
an ``id()`` or a weak reference again after each one (the identity check),
and the two million ``(action, time)`` records of student_8's exploded
sessions were read whole each time. Their reading is kept for the cell by a
fingerprint of what the column holds (``shape_witness``), taken again
without holding any of it: a column changed in place reads again
(`test_a_changed_column_is_read_again`, and the controls in
``test_notebook/restore/test_a_column_read_once_in_a_cell_is_checked_again_when_changed.py``).
"""

from __future__ import annotations

import pytest

from cash import _native
from cash.notebook.statement import identity_holders
from tests._cell_driver import run_cash_cell

pytest.importorskip("pandas")
pytestmark = pytest.mark.skipif(_native.kernels is None, reason=f"no C walk kernels: {_native.REASON}")

#: More sessions than `identity_holders._MAX_SCANNED`: read at C speed.
MAKE = (
    "import datetime\nimport pandas as pd\n"
    "t0 = datetime.datetime(2020, 1, 1)\n"
    "recs = [(f'u{s % 7}', [(f'a{a}', t0 + datetime.timedelta(seconds=s + a)) for a in range(3)])"
    " for s in range(12_000)]\n"
    "df = pd.DataFrame(recs, columns=['user', 'actions'])"
)
ADDS = "df['k1'] = 1\ndf['k2'] = 2\ndf['k3'] = 3\ndf['k4'] = 4"


def _count_reads(monkeypatch) -> dict[str, int]:
    """Count the reads (`_Walk._read_nested` at C speed) of the sessions
    column, a list of lists, and of ``recs``, a list of tuples: ``recs``
    holds the column's lists too, so it is checked with ``df``."""
    reads = {"column": 0, "recs": 0}
    real = identity_holders._NESTED_LEAVES

    def counting(items, *args):
        if len(items) >= 12_000:
            reads["column" if type(items[0]) is list else "recs"] += 1
        return real(items, *args)

    monkeypatch.setattr(identity_holders, "_NESTED_LEAVES", counting)
    return reads


def test_the_column_is_read_once_in_a_cell_of_four_statements(cash_magics, monkeypatch):
    reads = _count_reads(monkeypatch)
    run_cash_cell(cash_magics, MAKE)
    assert reads == {"column": 1, "recs": 1}
    run_cash_cell(cash_magics, ADDS)
    assert reads == {"column": 1, "recs": 1}


def test_the_column_is_not_read_again_by_the_next_cells(cash_magics, monkeypatch):
    reads = _count_reads(monkeypatch)
    run_cash_cell(cash_magics, MAKE)
    run_cash_cell(cash_magics, "df['k1'] = 1")
    run_cash_cell(cash_magics, "df['k2'] = 2")
    assert reads == {"column": 1, "recs": 1}


def test_a_changed_column_is_read_again(cash_magics, monkeypatch):
    reads = _count_reads(monkeypatch)
    run_cash_cell(cash_magics, MAKE)
    run_cash_cell(cash_magics, "df['k1'] = 1\nrecs[5][1].append(('b', t0))\ndf['k2'] = 2\ndf['k3'] = 3")
    # When made, and once after the change: the list changed is in both.
    assert reads == {"column": 2, "recs": 2}


def test_a_column_changed_in_a_later_cell_is_read_again(cash_magics, monkeypatch):
    reads = _count_reads(monkeypatch)
    run_cash_cell(cash_magics, MAKE)
    run_cash_cell(cash_magics, "recs[5][1].append(('b', t0))")
    run_cash_cell(cash_magics, "df['k1'] = 1")
    assert reads["column"] == 2


def test_a_reading_of_many_numbers_is_kept_for_its_cell_alone():
    identity_holders.forget_flat_reads()
    # Numbers as big as an address: each one read is kept.
    rows = [[(identity_holders._LOWEST_ADDRESS + i, 1)] for i in range(12_000)]
    identity_holders.identity_held_reason("rows", rows, {"rows": rows})
    kept = identity_holders._KEPT_SHAPE_NUMBERS
    assert any(len(numbers) > kept for numbers, _lists, _nodes in identity_holders._SHAPE_READS.values())
    identity_holders.end_cell_reads()
    assert all(len(numbers) <= kept for numbers, _lists, _nodes in identity_holders._SHAPE_READS.values())
