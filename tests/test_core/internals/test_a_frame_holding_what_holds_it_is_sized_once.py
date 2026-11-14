"""A list holding a frame whose object column holds the list again is sized
in bounded time, each object once.

The sizes of a frame's object cells were walked afresh for each cell: a
cell reaching back to the list sized its frames again, and those their
cells, without end (a hang on Windows, a recursion error swallowed per
level elsewhere). The walk now carries the objects it has counted into the
cells.
"""

from __future__ import annotations

import time

import pytest

from cash import sizing

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

pytestmark = [pytest.mark.core]


def _cyclic(n: int) -> list:
    owner = [pd.DataFrame({"a": [1, 2]})]
    for i in range(n):
        owner.append(pd.DataFrame({"o": [owner, i]}))
        cells = np.empty(2, dtype=object)
        cells[0] = {"k": owner}
        owner.append(pd.DataFrame({"o": cells}))
    return owner


@pytest.mark.parametrize("size", [sizing.memory_footprint, sizing.estimate_object_size])
def test_a_cycle_through_object_columns_is_sized_quickly(size):
    owner = _cyclic(6)
    start = time.perf_counter()
    for value in (owner, owner[1], owner[-1]):
        assert 0 < size(value) < 1_000_000
    assert time.perf_counter() - start < 5


def test_each_frame_of_the_cycle_is_counted_once():
    owner = _cyclic(6)
    whole = sizing.memory_footprint(owner)
    frames = sum(sizing.memory_footprint(frame) for frame in owner[:1])
    assert whole > frames
    assert whole < 13 * sizing.memory_footprint(owner[1])
    assert sizing.pandas_nbytes(owner[1]) is not None


def test_a_frame_of_records_is_sized_as_before():
    """The control: cells that hold their own records are each walked
    through, as before."""
    records = [{"x": list(range(100 + i))} for i in range(10)]
    frame = pd.DataFrame({"o": records, "n": range(10)})
    each = sum(sizing.memory_footprint(record) for record in records)
    index = int(frame.index.nbytes)
    assert sizing.pandas_nbytes(frame) == index + 80 + 8 * 10 + each


def test_a_record_every_cell_holds_is_counted_once():
    shared = {"x": list(range(100))}
    frame = pd.DataFrame({"o": [shared] * 10})
    assert sizing.pandas_nbytes(frame) == int(frame.index.nbytes) + 8 * 10 + sizing.memory_footprint(shared)
