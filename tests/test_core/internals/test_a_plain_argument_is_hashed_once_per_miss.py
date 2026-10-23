"""A miss on plain data hashes its arguments once, for the key.

The check that the body left its arguments alone hashed them again after
the body: a second full hash of a list of ten million ints, a second of a
3.5 s first call whose work took 0.09 s. Plain lists and tuples are compared
by the identities of what they hold (`_plain_data.identity_snapshot`), and
their leaves cannot change in place, so when no identity moved and every
other argument cannot change at all, the second hash proves nothing more.

That holds for the content the key was built from, and another thread can
change the list between the key build and the snapshot: the snapshot
counts while the watchers still give the key's digest, or while no other
thread runs. The tests below run as a process with no other thread; the
test run's own watchdog thread is not one the race is about.
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


@pytest.fixture
def one_thread(monkeypatch):
    monkeypatch.setattr(threading, "active_count", lambda: 1)


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
def test_a_miss_on_plain_data_does_not_hash_it_again_after_the_body(disk_cash, one_thread, build):
    rows = build()  # built here: a parameter's value is held for the whole run
    f = disk_cash.cache(lambda rows, scale=1: len(rows) * scale)
    with method_calls(PurityChecks, "checked_arguments_hash") as rehash:
        f(rows, scale=2)
    assert rehash.calls == 0


def test_a_miss_hashes_a_big_plain_list_about_once(disk_cash, one_thread):
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
def test_an_argument_identities_cannot_vouch_for_is_hashed_again(disk_cash, one_thread, arg):
    """Control: a dict or a bytearray can change with every identity in place."""
    f = disk_cash.cache(lambda x: 1)
    with method_calls(PurityChecks, "checked_arguments_hash") as rehash:
        f(arg)
    assert rehash.calls == 1


def test_an_int_subclass_beside_a_plain_list_is_hashed_again(disk_cash, one_thread):
    """Control: a subclass of int can carry state that changes."""
    f = disk_cash.cache(lambda rows, n: len(rows) + n)
    with method_calls(PurityChecks, "checked_arguments_hash") as rehash:
        f([1, 2, 3], Tagged(4))
    assert rehash.calls == 1


def test_with_another_thread_an_unwatched_list_is_hashed_again(disk_cash):
    """With another thread alive and no watcher on the list, nothing shows
    the snapshot is of the content the key holds: the hash runs."""
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
    assert rehash.calls == 1


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
