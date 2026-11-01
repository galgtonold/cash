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


def test_a_text_column_in_many_chunks_gives_the_numbers_it_spells():
    # After a ``merge`` the column came in thousands of chunks, and the two
    # patterns over all of them took 6 s: the values that could spell a
    # number are picked by length, then searched as one array.
    import pyarrow

    forget_flat_reads()
    chunk = [f"w{i}" for i in range(20)]
    chunks = [pyarrow.array(chunk, type=pyarrow.large_string()) for _ in range(5_000)]
    chunks[3] = pyarrow.array([*chunk, "12345678", "0x7f00aa12bc", "1" * 19], type=pyarrow.large_string())
    s = pd.Series(pd.arrays.ArrowExtensionArray(pyarrow.chunked_array(chunks)))
    parts = identity_holders._arrow_text_spelling_numbers(s)
    assert parts[0].tolist() == [12345678]
    assert parts[1:] == [0x7F00AA12BC]
    forget_flat_reads()


def test_only_whole_ascii_numbers_of_number_length_are_read():
    # Picked by masks, not patterns: other digits, a space, a bad hex digit
    # or one digit too many spell nothing, as before.
    texts = ["1234567", "999999999999999999", "1234567890123456789", "\u0661\u0662\u0663\u0664\u0665\u0666\u0667"]
    texts += ["12 34567", "0x12g456", "0XABCDEF", "0x1234", "abc"]
    s = pd.Series(texts * 300, dtype="string[pyarrow]")
    forget_flat_reads()
    parts = identity_holders._arrow_text_spelling_numbers(s)
    assert sorted(set(parts[0].tolist())) == [1234567, 999999999999999999]
    assert set(parts[1:]) == {0xABCDEF}
    forget_flat_reads()


def test_a_text_column_of_no_number_length_gives_no_numbers():
    forget_flat_reads()
    assert identity_holders._arrow_text_spelling_numbers(pd.Series(["a", "bc", "x" * 30])) == []
    forget_flat_reads()


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
    seen = walk.seen_all()
    assert id(sessions[5]) in seen and id(sessions[5][3]) in seen
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


def test_an_id_of_a_record_inside_a_list_of_records_is_still_found():
    from cash.notebook.statement.identity_holders import identity_held_reason

    sessions = [[(f"a{j}", j) for j in range(20)] for _ in range(300)]
    ns = {"sessions": sessions}
    assert identity_held_reason("sessions", sessions, ns) is None
    # The id of one of its own records, as an int: the ids of what the
    # nested read saw are still there to compare with.
    sessions[3].append(("own", id(sessions[9][2])))
    assert identity_held_reason("sessions", sessions, ns) is not None


def _frame_from_records(n: int = 2_500):
    ns: dict = {}
    exec(
        "import pandas as pd\n"
        f"logs = [(f'u{{s % 7}}', [(f'a{{a}}', s * 20 + a) for a in range(5)]) for s in range({n})]\n"
        "df = pd.DataFrame(logs, columns=['user', 'actions'])\n",
        ns,
    )
    return ns


def test_a_frame_holding_the_lists_of_a_wide_list_of_records_is_stored_with_it():
    from cash.notebook.shared_objects import GaveUp, share_group

    # Past the widest list read down and the most targets asked up the
    # collector's referrers: the frame was refused, and ran every time.
    ns = _frame_from_records()
    gave_up = GaveUp()
    holders, shared = share_group(["df"], ns, ns, gave_up=gave_up)
    assert set(holders) == {"logs"} and not shared


def test_a_frame_holding_the_lists_of_nothing_else_has_no_holder():
    from cash.notebook.shared_objects import GaveUp, share_group

    ns = _frame_from_records()
    ns["logs"] = [(user, list(actions)) for user, actions in ns["logs"]]
    holders, shared = share_group(["df"], ns, ns, gave_up=GaveUp())
    assert not holders and not shared


def test_a_list_of_records_in_a_frame_held_elsewhere_is_found():
    from cash.notebook.shared_objects import GaveUp, share_group

    ns = _frame_from_records()
    ns["logs"] = [(user, list(actions)) for user, actions in ns["logs"]]
    ns["one"] = ns["df"]["actions"].iloc[40]
    holders, shared = share_group(["df"], ns, ns, gave_up=GaveUp())
    assert set(holders) == {"one"} and not shared


def test_lists_of_records_holding_a_shared_object_are_walked_below():
    from cash.notebook.shared_objects import GaveUp, share_group

    ns = _frame_from_records()
    ns["logs"] = [(user, list(actions)) for user, actions in ns["logs"]]
    ns["df"]["actions"].iloc[40].append(("held", ns.setdefault("box", [1])))
    holders, shared = share_group(["df"], ns, ns, gave_up=GaveUp())
    assert set(holders) == {"box"} and not shared


def test_repeated_long_texts_are_searched_once_each():
    # 8,000 log lines of 1 KB repeated through an exploded column: joined
    # whole, 5 GB of text; each line is read once now.
    lines = [("x" * 1000) + f"{i}" for i in range(50)]
    column = lines * 2_000
    column[777] = hex(0x7F00_1234_5670)
    column[778] = "first line\n" + str(0x7F00_1234_5680)
    walk = _Walk({})
    walk._read_leaves(column, {str})
    assert {0x7F00_1234_5670, 0x7F00_1234_5680} <= walk.numbers
