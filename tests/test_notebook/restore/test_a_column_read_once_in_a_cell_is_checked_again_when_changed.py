"""The identity check keeps what it read of a big column for the rest of the
cell and the cells after it
(``test_notebook/internals/test_a_column_left_alone_by_each_statement_of_a_cell_is_read_once.py``),
and gives every statement the answer a fresh read gives.

A column changed in place -- through the frame, through the list it was
built from, through a name bound to one of its sessions in a later
statement or cell -- is read again, and an ``id()`` put there is found as
before; one whose holder went away is answered as before. The memo keeps
nothing alive: the frame is freed as soon as nothing else holds it, inside
the cell, and what the memo keeps past the cell is numbers.
"""

from __future__ import annotations

import gc
import warnings

import pytest

from cash import _native
from cash.notebook.statement import identity_holders
from tests._cell_driver import run_cash_cell

pytest.importorskip("pandas")
pytestmark = pytest.mark.skipif(_native.kernels is None, reason=f"no C walk kernels: {_native.REASON}")

MAKE = (
    "import datetime\nimport pandas as pd\n"
    "class P:\n    pass\np = P()\n"
    "t0 = datetime.datetime(2020, 1, 1)\n"
    "recs = [(f'u{s % 7}', [(f'a{a}', t0 + datetime.timedelta(seconds=s + a)) for a in range(3)])"
    " for s in range(12_000)]\n"
    "df = pd.DataFrame(recs, columns=['user', 'actions'])"
)

#: Cells run after ``MAKE``; whether the identity check refuses ``df``;
#: how often the sessions column is read: once after the change (before it,
#: the reading ``MAKE``'s check kept answers). Where an ``id()`` was put, the
#: value refuses on its own, and the share check's retry with anchors, which
#: would check the same value, is not asked.
CHANGES = {
    "an id appended through the frame": (
        ["df['k1'] = 1\ndf['actions'][5].append(('c', hex(id(p))))\ndf['k2'] = 2"],
        True,
        1,
    ),
    "an id appended through the list it was built from": (
        ["df['k1'] = 1\nrecs[5][1].append(('c', str(id(p))))\ndf['k2'] = 2"],
        True,
        1,
    ),
    "an id written over a record in place": (["df['k1'] = 1\nrecs[3][1][1] = ('c', id(p))\ndf['k2'] = 2"], True, 1),
    "a new alias in a later statement": (
        ["df['k1'] = 1\nalias = df['actions'][3]\nalias.append(('c', id(p)))\ndf['k2'] = 2"],
        True,
        1,
    ),
    "a new alias in a later cell": (
        ["df['k1'] = 1", "alias = df['actions'][3]\nalias.append(('c', id(p)))\ndf['k2'] = 2"],
        True,
        1,
    ),
    "a holder dropped": (["df['k1'] = 1\ndel recs\ndf['k2'] = 2\ndf['k3'] = 3"], False, 0),
    # The very records again, in the very list: nothing to read again.
    "a session given its own records again": (
        ["df['k1'] = 1\nrecs[3][1][:] = list(recs[3][1])\ndf['k2'] = 2"],
        False,
        0,
    ),
}


def _count_column_reads(monkeypatch) -> list[int]:
    """Count the reads of the sessions column, a list of lists, at C speed."""
    reads = [0]
    real = identity_holders._NESTED_LEAVES

    def counting(items, *args):
        if len(items) >= 12_000 and type(items[0]) is list:
            reads[0] += 1
        return real(items, *args)

    monkeypatch.setattr(identity_holders, "_NESTED_LEAVES", counting)
    return reads


@pytest.mark.parametrize("memo", [True, False], ids=["kept", "read fresh"])
@pytest.mark.parametrize("change", list(CHANGES))
def test_a_change_is_found_as_a_fresh_read_finds_it(cash_magics, monkeypatch, change, memo):
    """Kept for the cell or read fresh each time (today's answer), the same
    answer; kept, the column read again only when what it holds changed."""
    cells, refused, column_reads = CHANGES[change]
    if not memo:
        monkeypatch.setattr(identity_holders, "_SHAPE_WITNESS", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(cash_magics, MAKE)
        reads = _count_column_reads(monkeypatch)
        for cell in cells:
            run_cash_cell(cash_magics, cell)
    held = [w for w in caught if "IDENTITY-HELD" in str(w.message) and "'df'" in str(w.message)]
    assert bool(held) is refused
    if memo:
        assert reads[0] == column_reads


def test_the_memo_keeps_no_object_alive(cash_magics):
    run_cash_cell(cash_magics, MAKE + "\nimport weakref")
    run_cash_cell(
        cash_magics,
        "column = weakref.ref(next(b.values for b in df._mgr.blocks if b.values.dtype == object))\n"
        "df['k1'] = 1\n"
        "df['k2'] = 2\n"
        "kept = __import__('cash.notebook.statement.identity_holders', fromlist=['_']).__dict__['_SHAPE_READS'].copy()\n"
        "del df, recs\n"
        "__import__('gc').collect()\n"
        "freed = column() is None",
    )
    ns = cash_magics.shell.user_ns
    # The memo had the column's reading while the frame went away.
    assert ns["kept"]
    assert ns["freed"] is True
    # Past the cell, the memo keeps numbers alone: no object of the cell's.
    assert all(
        type(numbers) is frozenset and all(type(n) is int for n in (*witness, *numbers, lists))
        for witness, (numbers, lists, _nodes) in identity_holders._SHAPE_READS.items()
    )
    gc.collect()
