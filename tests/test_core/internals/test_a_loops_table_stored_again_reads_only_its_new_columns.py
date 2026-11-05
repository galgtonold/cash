"""A loop's table stored again reads only its new columns.

p28-elo-ratings: ``for k ...: for hfa ...: h, _ = elo.run_elo(played, ...)``
stores ``played.assign(elo_home=..., elo_away=..., p_home=...)`` on every
pass. ``played``'s seven columns were read three more times a pass, though
the frame memo vouched for them: by the RAM tier's digest of the table it
keeps uncopied (`frame_sharing.checked_digest`), by the chunk store's digest
of each buffer of the disk entry (`chunk_store.digests`), and the new
columns by the next call's argument hash too. Fifteen passes read 2.4 GB
for 60 MB of data.

Now the memo reads the table's new columns once, at the store, and every
other digest of those bytes comes from what it vouches for
(`block_digests`). The digests are the ones a full read gives: what the
disk entry holds is checked by reading it back in a fresh process's tiers.
"""

from __future__ import annotations

import pytest

from cash.backends import frame_sharing
from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.tiered_backend import TieredBackend
from cash.decorator import arg_hashing
from tests._work_counts import hashed_bytes

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

pytestmark = pytest.mark.skipif(
    not frame_sharing._checked_works(), reason="the RAM tier keeps tables uncopied under copy-on-write only"
)

#: Rows enough for four numeric columns over `frame_sharing.CHECKED_MIN_BYTES`.
ROWS = frame_sharing.CHECKED_MIN_BYTES // 32 * 3 // 2


@pytest.fixture(autouse=True)
def empty_memo():
    arg_hashing.FRAME_DIGESTS.clear()
    yield
    arg_hashing.FRAME_DIGESTS.clear()


def _base() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    teams = np.array([f"T{i:03d}" for i in range(400)], dtype=object)
    return pd.DataFrame(
        {
            "season": rng.integers(2000, 2050, ROWS),
            "date": pd.date_range("2000-01-01", periods=ROWS, freq="min"),
            "home": pd.array(teams[rng.integers(0, 400, ROWS)], dtype="str"),
            "home_goals": rng.random(ROWS),
            "away_goals": rng.random(ROWS),
        }
    )


def _tiers(folder) -> TieredBackend:
    return TieredBackend([InMemoryBackend(), FileBackend(str(folder), flush_interval=0)])


def _store(tiers: TieredBackend, key: str, value) -> None:
    metadata = {"execution_time": 50.0}
    tiers.set(key, value, metadata)
    tiers.backends[1]._writes.wait_all()
    assert "DISK" in metadata["storage"]


def test_each_pass_reads_its_new_column_once_and_the_disk_holds_what_was_stored(tmp_path):
    base = _base()
    arg_hashing.cow_frame_digest(base)  # the loop's input, hashed before the loop (its key)
    tiers = _tiers(tmp_path / "c")
    stored = {}
    new_column = ROWS * 8
    try:
        for k in range(4):  # a loop's passes
            derived = base.assign(p_home=base["home_goals"] * (k + 1))
            with hashed_bytes() as read:
                _store(tiers, f"k{k}", derived)
                assert arg_hashing.cow_frame_digest(derived) is not None  # the next call's argument hash
            stored[f"k{k}"] = derived.copy()
            if k:
                # p_home once, nothing of the base's columns; before, each
                # pass read the base's numbers again for each digest.
                assert read.bytes < 1.5 * new_column, (k, read.bytes, new_column)
    finally:
        tiers.shutdown()
    again = _tiers(tmp_path / "c")
    try:
        for key, expected in stored.items():
            _meta, back = again.get(key)
            pd.testing.assert_frame_equal(back, expected)
    finally:
        again.shutdown()


def test_a_column_written_past_pandas_between_two_stores_is_read_again(tmp_path):
    # The memo stops vouching for a column a handle wrote: the next store
    # reads it, and its disk entry holds the new bytes, not a chunk named by
    # the old content.
    base = _base()
    arg_hashing.cow_frame_digest(base)
    tiers = _tiers(tmp_path / "c")
    try:
        _store(tiers, "first", base.assign(p_home=base["home_goals"] * 2))
        base["home_goals"].array[: ROWS // 2] = -1.0  # past pandas, into the base's own block
        second = base.assign(p_home=base["home_goals"] * 3)
        expected = second.copy()
        with hashed_bytes() as read:
            _store(tiers, "second", second)
        assert read.bytes >= 2 * ROWS * 8  # home_goals and p_home, at least
    finally:
        tiers.shutdown()
    again = _tiers(tmp_path / "c")
    try:
        _meta, back = again.get("second")
    finally:
        again.shutdown()
    pd.testing.assert_frame_equal(back, expected)
    assert (back["home_goals"].to_numpy()[: ROWS // 2] == -1.0).all()
