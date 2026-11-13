"""A miss on big series over one shared index reads each one's bytes once.

A solar model's step takes seven series of a year of minutes built over
one ``times`` index, fresh objects on every call, and returns a frame of
about the same size. Each argument's digest read the shared index again:
seven reads of the same 4 MB, a third of the key. The row labels are now a
piece of the digest the frame memo keeps by their memory, as it keeps a
column's (`content_hashers.index_memory_key`): the second series over the
same unchanged index adds the bytes the first added, unread. The digest is
the one a full read gives.

The checks around the body (the snapshot before it, the hash after it)
answer from the same memo, so a miss reads the arguments once. The memo
still sees what it must: a change made in place, through pandas or through
an array handed out by ``.values`` or ``to_numpy()``, is caught: a hit
leaves the argument as the call did.

The result, a frame of more than a mapped chunk's bytes, is written to
disk alone and joins the RAM tier at its first hit.
"""

from __future__ import annotations

import warnings

import pytest

from cash import content_hashers
from cash.decorator import arg_hashing
from tests._work_counts import hashed_bytes

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

pytestmark = pytest.mark.skipif(
    not arg_hashing.is_cow_pandas(pd.Series([1.0])), reason="the frame memo runs under copy-on-write only"
)

ROWS = 200_000  # 1.6 MB a column: each one's part of a digest is a tree digest
ARGS = 7


@pytest.fixture(autouse=True)
def empty_memo():
    arg_hashing.FRAME_DIGESTS.clear()
    yield
    arg_hashing.FRAME_DIGESTS.clear()


@pytest.fixture
def times():
    return pd.date_range("2024-01-01", periods=ROWS, freq="min", tz="UTC")


def _series(times, seed: int) -> list:
    rng = np.random.default_rng(seed)
    return [pd.Series(rng.random(ROWS), index=times) for _ in range(ARGS)]


def _step(a, b, c, d, e, f, g):
    s = a + b
    return pd.DataFrame({"x": s, "y": c * d, "z": e - f, "w": g, "v": s * 2})


def _total(a, b, c, d, e, f, g):
    # A small result: the disk tier's digest of a big one runs on its
    # writer thread, which a count of bytes hashed would take in too.
    return float((a + b - c * d + e - f + g).sum())


def test_series_sharing_an_index_digest_as_a_full_read_does(times):
    series = _series(times, 0)
    digests = [arg_hashing.cow_frame_digest(s) for s in series]
    assert digests == [content_hashers.builtin_hash(s) for s in series]
    other = pd.Series(series[0].to_numpy(), index=times.tz_convert("Europe/Berlin"))
    assert arg_hashing.cow_frame_digest(other) == content_hashers.builtin_hash(other) != digests[0]


@pytest.mark.filterwarnings("ignore:DatetimeIndex.values")
@pytest.mark.parametrize("keep_handle", [True, False], ids=["handle-kept", "handle-dropped"])
def test_an_index_written_through_a_handed_array_is_read_again(times, keep_handle):
    first, second = _series(times, 0)[:2]
    arg_hashing.cow_frame_digest(first)  # the memo keeps the index's piece
    labels = times.values
    labels.setflags(write=True)
    labels[0] = labels[1]
    if not keep_handle:
        del labels
    assert arg_hashing.cow_frame_digest(second) == content_hashers.builtin_hash(second)


def test_a_miss_reads_each_argument_and_the_shared_index_once(disk_cash, times):
    step = disk_cash.cache(_total)
    step(*_series(times, 0))  # analysis and the first miss's one-off work
    series = _series(times, 1)
    with hashed_bytes() as read:
        step(*series)
    one = series[0].to_numpy().nbytes
    # Seven columns and one index: key, snapshot and after-body check together.
    assert read.bytes < (ARGS + 1) * one * 1.25, read.bytes / one


def test_a_hit_on_fresh_series_reads_the_shared_index_once(disk_cash, times):
    step = disk_cash.cache(_total)
    first = step(*_series(times, 2))
    series = _series(times, 2)  # new objects, the same content
    with hashed_bytes() as read:
        again = step(*series)
    assert again == first
    assert read.bytes < (ARGS + 1) * series[0].to_numpy().nbytes * 1.25


def test_a_big_frame_result_waits_on_disk_then_joins_ram_at_its_first_hit(disk_cash, times):
    step = disk_cash.cache(_step)
    ram = disk_cash.backend.backends[0]
    stored = step(*_series(times, 3))
    assert ram.entry_count() == 0
    assert step(*_series(times, 3)).equals(stored)
    assert ram.entry_count() == 1


def _changing(disk_cash, change):
    @disk_cash.cache
    def step(a, b):
        change(a)
        return a * b

    return step


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda a: a.__setitem__(a.index[0], -1.0), id="through-pandas"),
        pytest.param(lambda a: _write_handed(a.values), id="through-values"),
        pytest.param(lambda a: _write_handed(a.to_numpy()), id="through-to_numpy"),
    ],
)
def test_a_big_series_changed_in_place_is_caught(disk_cash, times, change):
    step = _changing(disk_cash, change)
    # The key reads both into the memo; the checks after the body ask it.
    a, b = _series(times, 4)[:2]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        step(a, b)
    assert a.iloc[0] == -1.0
    said = [str(w.message) for w in caught]
    assert any("changed 'a' in place" in text for text in said), said
    plain = _series(times, 4)[:2]
    change(plain[0])
    for _ in range(2):
        a, b = _series(times, 4)[:2]
        assert step(a, b).equals(plain[0] * plain[1])
        assert a.iloc[0] == -1.0  # run again, or written back by the hit


def _write_handed(array) -> None:
    """A library writing through the array pandas handed out read-only."""
    array.setflags(write=True)
    array[0] = -1.0
