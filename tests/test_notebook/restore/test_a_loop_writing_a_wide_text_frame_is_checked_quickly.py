"""A loop writing one column of a wide text frame per pass is checked for
values found by identity without reading every column's text again.

``for k in temp.columns: temp[k] = temp[k] + ...`` over 60 columns of
20,000 rows took 113 s under cash where plain Python takes 2 s: after each
pass, the identity check turned every column into new Python strings,
joined and searched them, and asked each number spelled there against every
mapping of the process. Arrow's text is now read by Arrow, a column of
Python strings read before is known by its very strings, and a number is
placed among the mappings by a binary search.
"""

from __future__ import annotations

import pytest

from cash.notebook.statement import identity_holders
from cash.notebook.statement.identity_holders import _inside, _Walk, forget_flat_reads

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")


def test_a_number_is_placed_among_sorted_mappings():
    ranges = [(100, 200), (300, 400), (1000, 5000)]
    assert _inside({50, 100, 199, 200, 350, 999, 4999, 5000}, ranges) == {100, 199, 350, 4999}


def test_an_arrow_text_column_gives_the_numbers_it_spells():
    s = pd.Series(["abc", "12345678", "0x7f00aa12bc", None, "123456789012345678901", "x12345678"])
    parts = identity_holders._arrow_text_spelling_numbers(s)
    assert parts[0].tolist() == [12345678]
    assert parts[1:] == [0x7F00AA12BC]


def test_a_column_of_numbers_is_not_read_as_text():
    assert identity_holders._arrow_text_spelling_numbers(pd.Series([1, 2])) is None


def test_python_strings_read_before_are_known_by_identity():
    forget_flat_reads()
    texts = [str(n) for n in range(10_000_000, 10_020_000)]
    first = _Walk({})
    assert first._read_flat(texts)
    again = _Walk({})
    assert identity_holders._flat_read_before(texts) == frozenset(first.numbers)
    assert again._read_flat(list(texts))
    assert again.numbers == first.numbers
    # Equal text in other strings is read again, not taken from the memo.
    assert identity_holders._flat_read_before([str(n) for n in range(10_000_000, 10_020_000)]) is None
    forget_flat_reads()
    assert identity_holders._flat_read_before(texts) is None
