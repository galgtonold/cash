"""A miss on plain data hashes its arguments once, for the key.

The check that the body left its arguments alone hashed them again after
the body: a second full hash of a list of ten million ints, a second of a
3.5 s first call whose work took 0.09 s. Plain lists and tuples are compared
by the identities of what they hold (`_plain_data.identity_snapshot`), and
their leaves cannot change in place, so when no identity moved and every
other argument cannot change at all, the second hash proves nothing more.

That holds for the content the key was built from, and another thread can
change the list between the key build and the body: the snapshot is taken
before the key's walk, so such a write shows as a moved identity. The tests
run with the test run's own threads alive, and cash's writer threads, as in
any process with a disk cache.
"""

from __future__ import annotations

import threading

import pytest

from cash import value_watch
from cash.decorator.purity_checks import PurityChecks
from tests._work_counts import hashed_bytes, method_calls

pytestmark = pytest.mark.core


class Tagged(int):
    """An int that can carry attributes."""


@pytest.mark.parametrize(
    "build",
    [
        lambda: list(range(200_000)),
        lambda: tuple(range(200_000)),
        lambda: [(i, str(i), float(i)) for i in range(50_000)],
        lambda: [[i, i + 1] for i in range(50_000)],
    ],
    ids=["list of ints", "tuple of ints", "list of tuples", "list of lists"],
)
def test_a_miss_on_plain_data_does_not_hash_it_again_after_the_body(disk_cash, build):
    rows = build()  # built here: a parameter's value is held for the whole run
    f = disk_cash.cache(lambda rows, scale=1: len(rows) * scale)
    with method_calls(PurityChecks, "checked_arguments_hash") as rehash:
        f(rows, scale=2)
    assert rehash.calls == 0


def test_a_miss_hashes_a_big_plain_list_about_once(disk_cash):
    rows = list(range(300_000))
    f = disk_cash.cache(lambda rows: len(rows))
    f([1])  # set-up work of a first call, not of this argument
    with hashed_bytes() as hashed:
        f(rows)
    # One hash of ~2.7 MB of pickled ints for the key; two was the old count.
    one = len(__import__("pickle").dumps(rows, protocol=5))
    assert hashed.bytes < 1.5 * one, (hashed.bytes, one)


@pytest.mark.parametrize(
    "arg",
    [{"a": [1, 2]}, bytearray(b"abc"), [bytearray(b"abc")], [{"a": 1}]],
    ids=["dict", "bytearray", "list of bytearrays", "list of dicts"],
)
def test_an_argument_identities_cannot_vouch_for_is_hashed_again(disk_cash, arg):
    """Control: a dict or a bytearray can change with every identity in place."""
    f = disk_cash.cache(lambda x: 1)
    with method_calls(PurityChecks, "checked_arguments_hash") as rehash:
        f(arg)
    assert rehash.calls == 1


def test_an_int_subclass_beside_a_plain_list_is_hashed_again(disk_cash):
    """Control: a subclass of int can carry state that changes."""
    f = disk_cash.cache(lambda rows, n: len(rows) + n)
    with method_calls(PurityChecks, "checked_arguments_hash") as rehash:
        f([1, 2, 3], Tagged(4))
    assert rehash.calls == 1


def test_with_another_thread_an_unwatched_list_is_hashed_once(disk_cash):
    """Another thread alive does not cost a second hash: the snapshot taken
    before the key's walk is of the content the key holds, whoever runs."""
    rows = list(range(200_000))
    stop = threading.Event()
    other = threading.Thread(target=stop.wait, daemon=True)
    other.start()
    try:
        f = disk_cash.cache(lambda rows: len(rows))
        with method_calls(PurityChecks, "checked_arguments_hash") as rehash:
            f(rows)
    finally:
        stop.set()
        other.join(10)
    assert rehash.calls == 0


def test_with_another_thread_a_watched_list_is_not_hashed_again(disk_cash, monkeypatch):
    """The watchers still giving the key's digest after the snapshot show it
    is of that content, whatever other threads run."""
    if not value_watch.enabled() or value_watch.VERIFY:
        pytest.skip("needs the watchers, not hashing behind each reuse")
    cw = value_watch._cw
    before = cw.stats()
    cw.reset()
    cw.configure(min_bytes=0)
    rows = list(range(200_000))
    ns = {"rows": rows}
    monkeypatch.setattr(value_watch, "_anchors", lambda: (ns,))
    stop = threading.Event()
    other = threading.Thread(target=stop.wait, daemon=True)
    other.start()
    try:
        f = disk_cash.cache(lambda rows, scale=1: len(rows) * scale)
        f(rows)  # the first hash only sights it; the second arms it
        f(rows, scale=3)
        with method_calls(PurityChecks, "checked_arguments_hash") as rehash:
            f(rows, scale=2)
    finally:
        stop.set()
        other.join(10)
        cw.reset()
        value_watch.reset()
        cw.configure(min_bytes=before["min_bytes"], budget=before["budget"])
    assert rehash.calls == 0


def _counted_snapshots(monkeypatch) -> list:
    from cash import _plain_data

    taken: list = []
    real = _plain_data.identity_snapshot

    def counted(value):
        taken.append(value)
        return real(value)

    monkeypatch.setattr(_plain_data, "identity_snapshot", counted)
    return taken


@pytest.fixture
def watched(monkeypatch):
    """The watchers track every list the namespace binds, however small."""
    if not value_watch.enabled() or value_watch.VERIFY:
        pytest.skip("needs the watchers, not hashing behind each reuse")
    cw = value_watch._cw
    before = cw.stats()
    cw.reset()
    cw.configure(min_bytes=0)
    ns: dict = {}
    monkeypatch.setattr(value_watch, "_anchors", lambda: (ns,))
    yield ns
    cw.reset()
    value_watch.reset()
    cw.configure(min_bytes=before["min_bytes"], budget=before["budget"])


def test_a_hit_on_watched_lists_takes_no_identity_snapshot(disk_cash, monkeypatch, watched):
    """A watched list's snapshot waits for the check before the body, which
    a hit does not run: a hit on a million ints paid 44 ms for one."""
    big, small = list(range(200_000)), list(range(1000))
    watched.update(big=big, small=small)
    f = disk_cash.cache(lambda a, b: len(a) + len(b))
    f(big, small)
    f(big, small)
    taken = _counted_snapshots(monkeypatch)
    for _ in range(3):
        assert f(big, small) == 201_000
    assert taken == []


def test_a_hit_on_a_tuple_of_numbers_reads_none_of_its_items_for_a_snapshot(disk_cash, monkeypatch, watched):
    """Whether a tuple can hold a list was read item by item on every key
    build, hits included: a hit on a million ints took twice as long. Once
    the collector untracked the tuple, it holds no list, and nothing is
    read; a tuple holding a list is still snapshotted."""
    import gc

    from cash.decorator import arg_hashing

    numbers, small = tuple(range(200_000)), list(range(1000))
    nested = (1, (2, [3]))
    watched.update(small=small)
    f = disk_cash.cache(lambda a, b: len(a) + len(b))
    f(numbers, small)
    f(nested, small)
    gc.collect()
    assert not gc.is_tracked(numbers) and gc.is_tracked(nested)
    scans: list = []

    class Counted(frozenset):
        def issuperset(self, items):
            scans.append(1)
            return super().issuperset(items)

    monkeypatch.setattr(arg_hashing, "_PRIM_SET", Counted(arg_hashing._PRIM_SET))
    taken = _counted_snapshots(monkeypatch)
    for _ in range(3):
        assert f(numbers, small) == 201_000
    assert scans == [] and taken == []
    f(nested, small)
    assert any(value is nested for value in taken)


@pytest.mark.parametrize("inside", ["a dict", "a bytearray"])
def test_a_tuple_holding_something_written_in_place_is_still_checked(disk_cash, inside):
    """A dict or bytearray inside a tuple can be untracked with it, and
    changed in place. The snapshot skipped for such a tuple compares
    lists only; what else it holds is hashed again after the body, as it
    was, so a write by the body is seen and a later call is not served."""
    import gc

    held = {"n": 1} if inside == "a dict" else bytearray(b"\x01")
    pair = (0, held)

    def bump(t):
        if isinstance(t[1], dict):
            t[1]["n"] += 1
            return t[1]["n"]
        t[1][0] += 1
        return t[1][0]

    f = disk_cash.cache(bump)
    gc.collect()
    assert [f(pair) for _ in range(3)] == [2, 3, 4]


def test_a_miss_on_a_watched_list_is_still_hashed_once(disk_cash, watched):
    rows = list(range(200_000))
    watched["rows"] = rows
    f = disk_cash.cache(lambda rows, scale=1: len(rows) * scale)
    f(rows)
    f(rows, scale=3)  # armed
    with method_calls(PurityChecks, "checked_arguments_hash") as rehash:
        f(rows, scale=2)
    assert rehash.calls == 0


def test_a_watched_list_written_after_its_arm_is_hashed_again(disk_cash, monkeypatch, watched):
    """A write between the arm read before the walk and the check: the
    snapshot taken at the check would stand for the new content, so the list
    is hashed again after the body instead."""
    rows = list(range(200_000))
    watched["rows"] = rows
    f = disk_cash.cache(lambda rows, scale=1: len(rows) * scale)
    f(rows)
    f(rows, scale=3)
    real = PurityChecks.argument_identities

    def written(self, *args, **kwargs):
        rows[5] = -5
        return real(self, *args, **kwargs)

    monkeypatch.setattr(PurityChecks, "argument_identities", written)
    with method_calls(PurityChecks, "checked_arguments_hash") as rehash:
        f(rows, scale=2)
    assert rehash.calls == 1


def test_a_tuple_of_ints_takes_no_snapshot_before_its_walk(disk_cash, monkeypatch):
    """Nothing in it can change: the snapshot would be ``[None]``."""
    rows = tuple(range(200_000))
    f = disk_cash.cache(lambda rows, n: len(rows) + n)
    taken = _counted_snapshots(monkeypatch)
    f(rows, 1)
    # Taken once, before the body, by the check of the body's writes alone.
    assert sum(value is rows for value in taken) == 1
