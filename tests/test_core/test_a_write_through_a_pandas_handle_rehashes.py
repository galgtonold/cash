"""A write through a handle pandas hands out is seen by the frame memo.

Under copy-on-write the memo trusts block identities: a frame only changes by
getting new block arrays. ``s.array`` (every dtype), and ``s.values`` or
``to_numpy()`` of an extension dtype, hand out the block's own array, and a
write through them lands in place with the identities unchanged. The memo
answered with the old content hash and the call was served 6.0 where the
series summed to 105.0.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from cash import Cash
from cash.backends import InMemoryBackend
from cash.decorator import arg_hashing

pytestmark = pytest.mark.skipif(
    not arg_hashing.is_cow_pandas(pd.Series([1.0])), reason="the frame memo runs under copy-on-write only"
)


@pytest.fixture
def total():
    cash = Cash(backend=InMemoryBackend(), register_magic=False)

    @cash.cache
    def total(value):
        return float(value.sum().sum()) if hasattr(value, "columns") else float(value.sum())

    return total


WRITES = {
    "float via array": (lambda: pd.Series([1.0, 2.0, 3.0]), lambda s: s.array.__setitem__(0, 100.0)),
    "float via array to_numpy": (
        lambda: pd.Series([1.0, 2.0, 3.0]),
        lambda s: s.array.to_numpy().__setitem__(0, 100.0),
    ),
    "Int64 via values": (lambda: pd.Series([1, 2, 3], dtype="Int64"), lambda s: s.values.__setitem__(0, 100)),
    "categorical via values": (
        lambda: pd.Series(["1", "2", "1"], dtype=pd.CategoricalDtype(["1", "2", "100"])),
        lambda s: s.values.__setitem__(0, "100"),
    ),
    "durations via array": (
        lambda: pd.Series(pd.to_timedelta([1, 2, 3], unit="s")),
        lambda s: s.array.__setitem__(0, pd.Timedelta(100, unit="s")),
    ),
}


def _sum(value):
    if value.dtype.kind == "m":
        return value.sum().total_seconds()
    if value.dtype.name == "category":
        return float(pd.to_numeric(value.astype(str)).sum())
    return float(value.sum())


@pytest.mark.parametrize("make, write", list(WRITES.values()), ids=list(WRITES))
def test_a_write_through_a_handle_is_seen(make, write):
    cash = Cash(backend=InMemoryBackend(), register_magic=False)

    @cash.cache
    def total(value):
        return _sum(value)

    series = make()
    assert total(series) == _sum(series)
    assert total(series) == _sum(series)  # the memo holds it now
    write(series)
    assert total(series) == _sum(series)


def test_a_write_through_a_frame_column_is_seen(total):
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    assert total(frame) == 10.0
    assert total(frame) == 10.0
    frame["a"].array[0] = 100.0
    assert float(frame.to_numpy().sum()) == 109.0
    assert total(frame) == 109.0


def test_arrow_strings_written_through_values_are_seen():
    """An Arrow array is written by swapping the array it holds, which the
    signature sees; its handle is not recorded, so this guards that."""
    pytest.importorskip("pyarrow")
    cash = Cash(backend=InMemoryBackend(), register_magic=False)

    @cash.cache
    def joined(s):
        return "".join(s.tolist())

    series = pd.Series(["a", "b"], dtype=pd.StringDtype("pyarrow"))
    assert joined(series) == "ab"
    series.values[0] = "z"
    assert joined(series) == "zb"


def test_an_untouched_frame_still_uses_the_memo(total, monkeypatch):
    """Positive control: reading and hashing without a writable handle keeps
    the memo, so the content is hashed once."""
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    assert total(frame) == 10.0
    calls = []
    real = arg_hashing.builtin_hash
    monkeypatch.setattr(arg_hashing, "builtin_hash", lambda v: calls.append(1) or real(v))
    frame.to_numpy()  # a read-only view: no handle to write through
    frame["a"].values  # the same
    assert total(frame) == 10.0
    assert calls == []
