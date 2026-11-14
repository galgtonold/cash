"""The numbers of an integer column that could be an ``id()`` are read with
one mask of the bits an address leaves clear. Read with ``%`` and a boolean
index, a column of 8 million numbers took the share check's whole 0.1 s, so
whether its frame was stored came down to a few milliseconds of CPU time.
The mask must keep exactly the numbers the plain rule keeps: at or above
the lowest address, below the highest, and a multiple of a pointer's size."""

from __future__ import annotations

import pytest

from cash.notebook.statement import identity_holders
from cash.notebook.statement.identity_holders import identity_held_reason

np = pytest.importorskip("numpy")


def _plain(array):
    low, high, step = identity_holders._LOWEST_ADDRESS, identity_holders._HIGHEST_ADDRESS, identity_holders._ALIGNMENT
    return [int(n) for n in array.ravel().tolist() if low <= int(n) < high and int(n) % step == 0]


_LOW = 1 << 22
_HIGH = 1 << 48


@pytest.mark.parametrize(
    "array",
    [
        np.array([_LOW - 8, _LOW, _LOW + 4, _LOW + 8, _HIGH - 8, _HIGH, _HIGH + 8, -8, -_LOW, 2**63 - 8], np.int64),
        np.array([_LOW, _HIGH - 8, _HIGH, 2**63, 2**64 - 8], np.uint64),
        np.array([_LOW, _LOW + 8, 3], dtype=">i8"),
        (np.arange(24, dtype=np.int64).reshape(4, 6) * (_LOW + 8))[:, ::2],
        np.random.default_rng(0).integers(-(2**63), 2**63 - 1, 50_000, dtype=np.int64),
        np.random.default_rng(1).integers(0, 2**51, 50_000, dtype=np.int64) * 8,
    ],
    ids=["edges", "unsigned", "big-endian", "strided", "any-int64", "aligned"],
)
def test_the_mask_keeps_what_the_rule_keeps(array):
    found = identity_holders._addresses_in(array)
    assert found.dtype == np.int64
    assert found.tolist() == _plain(array)


def test_an_id_in_a_big_column_is_still_found():
    held = object()
    column = np.arange(2_000_000, dtype=np.int64) * 12
    column[1_234_567] = id(held)
    reason = identity_held_reason("column", column, {"column": column, "held": held})
    assert reason is not None
    assert "id()" in str(reason)
