"""A table of Python strings written to disk again reads only its new columns.

Without pyarrow, pandas 3 keeps text as Python strings, and the RAM tier
does not keep such a table uncopied: p28's loop results (``h, _ =
run_elo(played, ...)``, 15 passes) went to disk from the caller's table,
copied into fresh bytes, and the chunk store read every buffer of every
entry again, though they were ``played``'s columns: 922 MB a grid pass.

Now a block whose bytes the frame memo vouches for is copied into an
array named by its content key (`block_digests.note_own`), and a text
column's UTF-8 buffers, kept for its content (`text_columns`), are bytes
nothing can change, whose digest is read once. What the disk holds is
checked by reading it back, and so is a column written past pandas
between two writes.

The frame memo runs under copy-on-write only: pandas 2 without it lets any
handle write a block, so nothing vouches for the base's bytes and each
write reads them again (what is read is pinned under copy-on-write only).
What is stored anew and what the disk holds are pinned on either.
"""

from __future__ import annotations

import pytest

from cash.backends.file_backend import FileBackend
from cash.decorator import arg_hashing
from tests._work_counts import hashed_bytes

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

ROWS = 300_000  # each column over a chunk (`chunk_store.CHUNK_MIN_BYTES`)


@pytest.fixture(autouse=True)
def empty_memo():
    arg_hashing.FRAME_DIGESTS.clear()
    yield
    arg_hashing.FRAME_DIGESTS.clear()


def _base() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    teams = np.array([f"T{i:03d} ü" for i in range(400)], dtype=object)
    return pd.DataFrame(
        {
            "season": rng.integers(2000, 2050, ROWS),
            "date": pd.date_range("2000-01-01", periods=ROWS, freq="min"),
            "home": pd.array(teams[rng.integers(0, 400, ROWS)], dtype=pd.StringDtype("python", na_value=np.nan)),
            "id": pd.array([f"match {i}" for i in range(ROWS)], dtype=pd.StringDtype("python")),
            "goals": rng.random(ROWS),
        }
    )


def _chunk_bytes(folder) -> int:
    chunks = folder / "chunks"
    return sum(p.stat().st_size for p in chunks.iterdir()) if chunks.exists() else 0


def _write(backend: FileBackend, key: str, value) -> None:
    backend.set(key, value, {"execution_time": 50.0})
    backend._writes.wait_all()


def test_each_write_reads_its_new_column_once_and_the_disk_holds_what_was_written(tmp_path):
    base = _base()
    memo = arg_hashing.is_cow_pandas(base)
    arg_hashing.cow_frame_digest(base)  # the loop's input, hashed before the loop (its key)
    backend = FileBackend(str(tmp_path / "c"), flush_interval=0)
    written = {}
    new_column = ROWS * 8
    for k in range(4):  # a loop's passes
        derived = base.assign(p_home=base["goals"] * (k + 1))
        before = _chunk_bytes(tmp_path / "c")
        with hashed_bytes() as read:
            _write(backend, f"k{k}", derived)
        written[f"k{k}"] = derived.copy()
        if k:
            # p_home once, nothing of the base's columns, text or not; and
            # only p_home is stored anew.
            if memo:
                assert read.bytes < 1.5 * new_column, (k, read.bytes, new_column)
            assert _chunk_bytes(tmp_path / "c") - before < 1.1 * new_column
    again = FileBackend(str(tmp_path / "c"), flush_interval=0)
    for key, expected in written.items():
        back = again.get(key)[1]
        pd.testing.assert_frame_equal(back, expected)
        assert repr(back["id"].dtype) == repr(expected["id"].dtype)


def test_a_text_column_written_past_pandas_between_two_writes_is_written_anew(tmp_path):
    base = _base()
    arg_hashing.cow_frame_digest(base)
    backend = FileBackend(str(tmp_path / "c"), flush_interval=0)
    _write(backend, "first", base.assign(p_home=base["goals"] * 2))
    base["home"].array[: ROWS // 2] = "changed"  # past pandas, into the base's own block
    base["id"].array[3] = None
    second = base.assign(p_home=base["goals"] * 3)
    expected = second.copy()
    _write(backend, "second", second)
    back = FileBackend(str(tmp_path / "c"), flush_interval=0).get("second")[1]
    pd.testing.assert_frame_equal(back, expected)
    assert (back["home"].iloc[: ROWS // 2] == "changed").all()
    assert back["id"].iloc[3] is pd.NA
