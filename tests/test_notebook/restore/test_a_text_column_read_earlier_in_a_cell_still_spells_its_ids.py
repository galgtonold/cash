"""A big column of text the identity check read earlier in the cell gives
every number it spells when it is checked again, those the first check had
met elsewhere before it too.

The first read kept only the numbers new to its walk: ``str(id(p))`` met
beside the column, above it, was left out of what the column was
kept as spelling, and a later check of the column alone, answered from
that, found no ``id()`` and let the value be restored on its own.
"""

from __future__ import annotations

from cash.notebook.statement.identity_holders import forget_flat_reads, identity_held_reason


class P:
    pass


def test_the_column_alone_still_spells_the_id():
    p = P()
    key = str(id(p))
    column = [f"w{i}" for i in range(20_000)] + [key]
    ns = {"p": p, "column": column}
    forget_flat_reads()
    try:
        # The text beside the column is a level above it: met before it.
        pair = (key, [column])
        assert identity_held_reason("pair", pair, {**ns, "pair": pair}) is not None
        assert identity_held_reason("column", column, ns) is not None
    finally:
        forget_flat_reads()
