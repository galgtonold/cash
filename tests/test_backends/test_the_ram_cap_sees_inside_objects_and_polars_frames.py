"""The RAM tier's byte cap counts what an object or a polars frame holds.

Only builtin containers, pandas frames, arrays and sparse matrices were
looked into; anything else counted as its own ``sys.getsizeof``. A 16 MB
polars frame or dataclass holding an array was 48 bytes, so the cap never
evicted them and the process grew without bound.
"""

from __future__ import annotations

import dataclasses

import pytest

np = pytest.importorskip("numpy")

from cash.backends.memory_backend import InMemoryBackend
from cash.sizing import memory_footprint

MB = 1024**2


@dataclasses.dataclass
class _Fit:
    coef: object
    score: float


class _Slotted:
    __slots__ = ("coef", "__private")

    def __init__(self, coef):
        self.coef = coef
        self.__private = np.zeros(1000)


def _kept(make, n=12, size=2 * MB, cap=8 * MB) -> int:
    """How many of *n* values of *size* bytes a RAM tier capped at *cap* keeps."""
    ram = InMemoryBackend(max_size_bytes=cap)
    for i in range(n):
        ram.set(f"k{i}", make(np.full(size // 8, float(i))))
    return ram.entry_count()


def test_an_object_holding_an_array_is_counted_through_its_attributes():
    coef = np.zeros(1_000_000)
    assert memory_footprint(_Fit(coef, 1.0)) >= coef.nbytes
    assert memory_footprint(_Slotted(coef)) >= coef.nbytes + 8000
    assert memory_footprint([_Fit(np.zeros(10_000), 0.0) for _ in range(200)]) >= 200 * 80_000


def test_a_polars_frame_is_counted_by_its_data():
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({"x": np.zeros(1_000_000)})
    assert memory_footprint(frame) >= 8_000_000
    assert memory_footprint({"variables": {"df": frame}}) >= 8_000_000


@pytest.mark.parametrize("kind", ["array", "object", "polars"])
def test_the_cap_evicts_objects_and_polars_frames_like_arrays(kind):
    if kind == "polars":
        pl = pytest.importorskip("polars")
        make = lambda a: pl.DataFrame({"x": a})  # noqa: E731 - one-line factories, one per kind
    elif kind == "object":
        make = lambda a: _Fit(a, float(a[0]))  # noqa: E731 - one-line factories, one per kind
    else:
        make = lambda a: a  # noqa: E731 - one-line factories, one per kind
    assert _kept(make) <= 4


def test_a_frame_column_of_nested_records_counts_what_the_cells_hold():
    pd = pytest.importorskip("pandas")
    rows = [[{"id": i, "tags": list(range(20))}] for i in range(5_000)]
    # Each cell: a list holding a dict holding a list of 20 ints, well over 400 bytes.
    assert memory_footprint(pd.DataFrame({"t": rows})) >= 5_000 * 400


def test_a_long_list_of_records_is_sized_from_a_sample():
    many = [_Fit(np.zeros(16), float(i)) for i in range(100_000)]
    exact = sum(memory_footprint(f) for f in many[:1000]) * 100
    assert 0.8 * exact <= memory_footprint(many) <= 1.2 * exact


def test_an_attribute_every_record_shares_is_counted_once():
    shared = np.zeros(2_000_000)
    many = [_Fit(shared, float(i)) for i in range(100_000)]
    assert shared.nbytes <= memory_footprint(many) < shared.nbytes + 100_000 * 1_000
