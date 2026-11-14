"""A frame, series or object array whose cells hold what holds it has no
key, and finding that out takes no time.

An object column is keyed by a walk of its own over its cells. When a cell
held the list holding the frame, that walk met the frame again and keyed
it afresh, down to Python's recursion limit and then again through every
fallback: a statement or call reading such a list never finished. The
frame is now refused as a value that contains itself is (a TypeError, so
the call runs uncached).
"""

from __future__ import annotations

import time

import pytest

from cash.canonical_form import CyclicValueError, canonical_bytes
from cash.content_hashers import BUILTIN_CONTENT, hash_pandas

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

pytestmark = [pytest.mark.core]


def _through_a_column():
    owner = [pd.DataFrame({"a": [1, 2]})]
    owner.append(pd.DataFrame({"o": [owner, 1]}))
    return owner


def _through_a_dict_in_a_column():
    owner = [pd.DataFrame({"a": [1, 2]})]
    owner.append(pd.DataFrame({"o": [{"k": owner}, 1]}))
    return owner


def _through_an_object_array():
    owner = [1]
    cells = np.empty(2, dtype=object)
    cells[0] = owner
    owner.append(cells)
    return owner


@pytest.mark.parametrize("build", [_through_a_column, _through_a_dict_in_a_column, _through_an_object_array])
def test_the_list_and_what_it_holds_are_refused_quickly(build):
    owner = build()
    start = time.perf_counter()
    for value in (owner, owner[-1]):
        with pytest.raises(CyclicValueError):
            canonical_bytes(value, BUILTIN_CONTENT)
    assert time.perf_counter() - start < 5


def test_a_frame_holding_another_list_is_keyed_by_its_content():
    """The control: a cell holding a list that holds nothing of the frame."""
    one = pd.DataFrame({"o": [[1, 2], 1]})
    two = pd.DataFrame({"o": [[1, 2], 1]})
    assert hash_pandas(one) == hash_pandas(two) is not None
    assert canonical_bytes([one], BUILTIN_CONTENT) == canonical_bytes([two], BUILTIN_CONTENT)
