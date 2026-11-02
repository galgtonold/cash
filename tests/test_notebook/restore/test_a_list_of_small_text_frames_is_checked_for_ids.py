"""A list of thousands of small text frames is checked for a number that
could be an ``id()`` as a big frame is.

``df_array.append(pd.DataFrame({...}))`` once per session, then
``pd.concat(df_array)``: the identity check read every small frame's text
through Arrow, several calls per column at a fixed cost each, after making
a column and a ``dtypes`` series for each, and the loop ran three times as
long as plain Python. A short Arrow column is now read in Python; what it
finds must be what Arrow finds in the same text.
"""

from __future__ import annotations

import pytest

from cash.notebook.statement.identity_holders import identity_held_reason, library_parts

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pytest.importorskip("pyarrow")

#: pandas 3's ``str``, Arrow text; pandas 2 keeps text as Python strings unless asked.
STR = pd.StringDtype("pyarrow", na_value=np.nan)

TEXTS = [
    "abc",
    "12345678",
    "0000012345678",
    "123456",  # too short
    "1234567890123456789",  # too long
    "0x7f00aa12bc",
    "0X7F00AA12BC",
    "0x1234",  # too short
    "x12345678",
    "12345678\n",  # a last newline is not the end of the text
    "12345678\n87654321",
    "\r12345678",
    "12345678 ",
    "١٢٣٤٥٦٧٨",  # Arabic-Indic digits
    None,
]


def _numbers(parts) -> list[int]:
    found: list[int] = []
    for part in parts:
        if isinstance(part, np.ndarray):
            found.extend(int(n) for n in part)
        elif isinstance(part, int):
            found.append(part)
    return sorted(found)


def _spelled(column) -> list[int]:
    return _numbers(part for piece in library_parts(column) for part in (piece if isinstance(piece, list) else [piece]))


def test_a_short_text_column_spells_what_a_long_one_does():
    short = pd.Series(TEXTS, dtype=STR)
    long = pd.Series(TEXTS * 20, dtype=STR)
    assert _spelled(short) == [12345678, 12345678, 0x7F00AA12BC, 0x7F00AA12BC]
    assert _spelled(long) == sorted(_spelled(short) * 20)


@pytest.mark.parametrize(
    "texts, found",
    [
        (["abc", "\u00e9" * 3 + "1234567"], [0]),  # not ASCII: read one by one
        (["a", "bc", "x" * 30, None], []),  # no value of a number's length: nothing at all
        (["\u00e9" * 4, "abc"], [0]),  # 8 bytes, as Arrow counts: looked through, an empty array
    ],
)
def test_a_short_text_column_is_read_as_a_long_one_is(texts, found):
    # Each part is one object the walk counts, so a short column gives what
    # a long one gives, an empty array or nothing at all.
    short = library_parts(pd.Series(texts, dtype=STR))[0]
    long = library_parts(pd.Series(texts * 20, dtype=STR))[0]
    assert [len(part) for part in short] == [len(part) for part in long] == found


def test_an_id_in_one_of_many_small_frames_is_found():
    held = object()
    frames = [pd.DataFrame({"user": "u1", "at": ["(1/2/2020)", "(3/4/2020)"], "what": ["a", "b"]}) for _ in range(300)]
    ns = {"df_array": frames, "held": held}
    assert identity_held_reason("df_array", frames, ns) is None
    frames[217] = pd.DataFrame({"user": "u1", "at": [str(id(held))], "what": ["a"]})
    assert identity_held_reason("df_array", frames, ns) is not None
    frames[217] = pd.DataFrame({"user": "u1", "at": ["x"], "what": [hex(id(held))]})
    assert identity_held_reason("df_array", frames, ns) is not None


def test_an_id_in_a_small_frames_column_names_is_found():
    held = object()
    frames = [pd.DataFrame({"user": ["u1"]}) for _ in range(50)]
    frames.append(pd.DataFrame({"user": ["u1"], str(id(held)): ["a"]}))
    assert identity_held_reason("df_array", frames, {"df_array": frames, "held": held}) is not None
