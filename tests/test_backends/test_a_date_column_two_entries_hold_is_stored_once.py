"""A column of dates or durations two entries hold is written to disk once.

numpy pickles a ``datetime64`` or ``timedelta64`` array into the stream: its
dtype exports no buffer, so protocol 5 cannot hand the data out of band, and
the chunk store never saw it. Each result of ``h, _ = run(played, k)`` in a
loop wrote ``played``'s date column into its entry again (7 MB a result for
900k rows), where every other column of ``played`` was stored once. These
tests pin that such a column now goes out of band, so a second entry holding
it adds no bytes for it, and that it comes back exactly: values, ``NaT``,
dtype, byte order and memory layout.
"""

from __future__ import annotations

import pickle

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from cash.backends.file_backend import FileBackend
from cash.backends.serialization import OUT_OF_BAND_MIN_BYTES, PickleSerializer

ROWS = 400_000  # 3.2 MB per 8-byte column: above the chunk store's minimum


def _backend(tmp_path):
    return FileBackend(str(tmp_path / "cache"), flush_interval=0)


def _store(backend, key, value):
    backend.set(key, value, {"execution_time": 1.0})
    backend._writes.wait_all()


def _folder_bytes(tmp_path):
    return sum(p.stat().st_size for p in (tmp_path / "cache").rglob("*") if p.is_file())


def _frame():
    dates = pd.Series(pd.date_range("2000-01-01", periods=ROWS, freq="min"))
    dates.iloc[7] = pd.NaT
    return pd.DataFrame({"date": dates, "x": np.arange(ROWS, dtype="float64")})


@pytest.mark.parametrize(
    "array",
    [
        np.arange(ROWS).astype("datetime64[ns]"),
        np.arange(ROWS).astype("timedelta64[s]"),
        np.arange(ROWS).astype(">M8[us]"),
        np.asfortranarray(np.arange(ROWS).astype("M8[ms]").reshape(-1, 400)),
    ],
    ids=["datetime64", "timedelta64", "big-endian", "fortran-order"],
)
def test_a_date_array_goes_out_of_band_and_comes_back_exactly(array):
    array = array.copy(order="K")
    array.flat[3] = np.datetime64("NaT") if array.dtype.kind == "M" else np.timedelta64("NaT", "s")
    stream, buffers = PickleSerializer().serialize_split(array)
    assert len(stream) < OUT_OF_BAND_MIN_BYTES
    assert sum(len(b) for b in buffers) == array.nbytes
    back = PickleSerializer().deserialize_split(stream, buffers)
    assert back.dtype == array.dtype
    assert back.shape == array.shape
    assert back.flags.f_contiguous == array.flags.f_contiguous
    assert back.tobytes(order="A") == array.tobytes(order="A")
    # The bytes one pickle writes, with no buffer callback, load the same.
    whole = pickle.loads(PickleSerializer().serialize(array))
    assert whole.dtype == array.dtype and whole.tobytes(order="A") == array.tobytes(order="A")


def test_a_short_or_strided_date_array_is_pickled_as_numpy_does():
    short = np.arange(100).astype("datetime64[s]")
    strided = np.arange(ROWS).astype("datetime64[s]")[::2]
    for array in (short, strided):
        stream, buffers = PickleSerializer().serialize_split(array)
        assert buffers == []
        back = PickleSerializer().deserialize_split(stream, buffers)
        assert back.dtype == array.dtype and np.array_equal(back, array)


def test_a_frame_with_a_date_column_comes_back_as_it_was_stored(tmp_path):
    frame = _frame()
    backend = _backend(tmp_path)
    _store(backend, "k1", frame)
    _, back = _backend(tmp_path).get("k1")
    pd.testing.assert_frame_equal(back, frame)


def test_a_date_column_two_entries_hold_is_written_once(tmp_path):
    frame = _frame()
    backend = _backend(tmp_path)
    _store(backend, "k1", frame)
    first = _folder_bytes(tmp_path)
    # Each pass of a loop returning its input's columns with one added.
    _store(backend, "k2", frame.assign(y=np.ones(ROWS)))
    added = _folder_bytes(tmp_path) - first
    one_column = ROWS * 8
    assert added < one_column * 1.5, f"the second entry added {added} bytes, more than its one new column"
