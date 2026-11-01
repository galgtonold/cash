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


def test_an_arrow_column_read_before_is_not_read_again():
    forget_flat_reads()
    s = pd.Series(["12345678", "x"])
    first = identity_holders._arrow_text_spelling_numbers(s)
    assert identity_holders._arrow_text_spelling_numbers(s) is first
    # A changed column is a new Arrow array: read again.
    s2 = s.copy()
    s2[1] = "87654321"
    assert sorted(identity_holders._arrow_text_spelling_numbers(s2)[0].tolist()) == [12345678, 87654321]
    forget_flat_reads()
    assert identity_holders._arrow_text_spelling_numbers(s) is not first


def test_lists_of_records_are_read_a_level_at_a_time():
    sessions = [[(f"a{j}", j) for j in range(20)] for _ in range(300)]
    sessions[-1].append(("c", 0x7F00_1234_5670))
    sessions[0].append((hex(0x7F00_1234_5680), 0))
    walk = _Walk({})
    assert walk._read_flat(sessions)
    assert {0x7F00_1234_5670, 0x7F00_1234_5680} <= walk.numbers
    # Each list and record is seen, as a walk would see it; a list counts
    # against the budget.
    assert id(sessions[5]) in walk.seen and id(sessions[5][3]) in walk.seen
    assert walk.budget == _Walk({}).budget - 300


def test_lists_holding_anything_but_records_are_walked():
    sessions = [[("a", 1)] for _ in range(300)]
    sessions[7].append({"k": 1})
    walk = _Walk({})
    assert not walk._read_flat(sessions)
    assert not walk.seen


def test_only_values_in_nested_tuples_have_no_identity_to_find():
    from cash.notebook.shared_objects import _values_in_tuples, exact_value_types

    exact = exact_value_types()
    assert _values_in_tuples([("a", 1), ("b", (2.0, None)), "c"], exact)
    assert not _values_in_tuples([("a", 1), ("b", [2])], exact)
    assert not _values_in_tuples([("a", {"k": 1})], exact)


def test_lists_of_records_are_each_found_by_identity():
    from cash.notebook.shared_objects import VALUE_TYPES, _held_identities

    sessions = [[(f"a{j}", j) for j in range(20)] for _ in range(300)]
    # Held elsewhere too (as ``pd.DataFrame(logs)`` holds the lists of
    # ``logs``), so each list is looked at rather than *sessions* alone.
    elsewhere = sessions[5]
    found = dict(_held_identities(sessions, VALUE_TYPES))
    assert {id(sessions), *map(id, sessions)} == set(found)
    # A list inside one is found too, walked as before.
    sessions[4].append([1])
    assert id(sessions[4][-1]) in dict(_held_identities(sessions, VALUE_TYPES))
    assert elsewhere is sessions[5]
