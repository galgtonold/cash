"""A copy the RAM tier keeps for the next hit goes to one caller only.

The tier keeps the copy a hit hands out and gives that very object to the
next hit when only cash holds it. While the first hit is still on its way
out of the tier, cash's is the only reference there is: a second thread's
hit arriving then must not be handed the same object (two callers writing
one result), and the kept copy being let go then (a new store, a delete,
the trim) must not fail the first hit.

The second thread is run at the exact point the first hit lets go of the
tier's lock after keeping its copy, so nothing here depends on timing.
"""

from __future__ import annotations

import pickle
import random
import threading

import pytest

from cash import value_watch
from cash.backends import memory_backend
from cash.backends.memory_backend import InMemoryBackend

pytestmark = pytest.mark.skipif(
    not value_watch.enabled(),
    reason="a copy is kept only where the C extension's watchers prove it unchanged",
)


def trie(n=3000, seed=5):
    rnd = random.Random(seed)
    root: dict = {}
    for _ in range(n):
        node = root
        for ch in "".join(rnd.choice("abcdefgh") for _ in range(rnd.randint(3, 9))):
            node = node.setdefault(ch, {})
        node["$"] = True
    return root


PLAIN = trie()


class _LockWithGap:
    """The tier's lock; once armed, the next release that leaves it free
    runs *at_gap* on another thread and waits for it."""

    def __init__(self, lock, at_gap):
        self._lock = lock
        self._at_gap = at_gap
        self._depth = 0
        self.armed = False
        self.ran = 0
        self.errors: list[BaseException] = []

    def __enter__(self):
        self._lock.acquire()
        self._depth += 1
        return self

    def __exit__(self, *exc):
        self._depth -= 1
        fire = self.armed and self._depth == 0
        if fire:
            self.armed = False
        self._lock.release()
        if fire:
            self.ran += 1
            other = threading.Thread(target=self._run)
            other.start()
            other.join()
        return False

    def _run(self):
        try:
            self._at_gap()
        except BaseException as exc:  # noqa: BLE001 - shown by the test
            self.errors.append(exc)


def _gap_after_keeping(monkeypatch, memory: InMemoryBackend, at_gap) -> _LockWithGap:
    """Run *at_gap* on another thread right after the next hit of *memory*
    that keeps its copy lets go of the lock."""
    lock = _LockWithGap(memory._lock, at_gap)
    monkeypatch.setattr(memory, "_lock", lock)
    real_trim = memory._trim_handed

    def trim():
        real_trim()
        lock.armed = True  # the copy is kept: the release after this is the gap

    monkeypatch.setattr(memory, "_trim_handed", trim)
    return lock


@pytest.fixture
def memory():
    backend = InMemoryBackend()
    backend.set("k", trie(), {"size": 100_000})
    assert type(backend._store["k"][1]) is memory_backend._Marshalled, "kept as marshal bytes"
    return backend


def test_a_hit_arriving_while_the_first_is_handed_its_copy_gets_its_own(memory, monkeypatch):
    seen: list = []
    lock = _gap_after_keeping(monkeypatch, memory, lambda: seen.append(memory.get("k")[1]))

    _meta, first = memory.get("k")

    assert lock.ran == 1 and not lock.errors, lock.errors
    assert seen[0] == PLAIN and first == PLAIN
    assert seen[0] is not first, "two threads were handed one object"
    first["mine"] = 1
    assert "mine" not in seen[0]


def test_the_copy_is_still_handed_back_once_its_caller_let_go(memory, monkeypatch):
    """The positive control: the kept copy is reused when nobody holds it."""
    _meta, first = memory.get("k")
    first_id = id(first)
    del first
    _meta, second = memory.get("k")
    assert id(second) == first_id


@pytest.mark.parametrize("let_go", ["delete", "store again", "clear"])
def test_a_kept_copy_let_go_while_it_is_handed_out_still_reaches_its_caller(memory, monkeypatch, let_go):
    def at_gap():
        if let_go == "delete":
            memory.delete("k")
        elif let_go == "store again":
            memory.set("k", {"other": 1}, {"size": 10})
        else:
            memory.clear()

    lock = _gap_after_keeping(monkeypatch, memory, at_gap)

    _meta, first = memory.get("k")  # raised IndexError

    assert lock.ran == 1 and not lock.errors, lock.errors
    assert first == PLAIN


def test_a_kept_copy_trimmed_at_once_still_reaches_its_caller(memory, monkeypatch):
    """A copy over the tier's budget for kept copies is let go by the same
    hit that kept it."""
    real_trim = memory._trim_handed

    def trim():
        real_trim()
        memory._release_handed("k")

    monkeypatch.setattr(memory, "_trim_handed", trim)
    _meta, first = memory.get("k")
    assert first == PLAIN


class Trie:
    def __init__(self):
        self.root = trie()


def _read_from_disk(memory: InMemoryBackend):
    """Hand *memory* a value as the tiered backend hands it one just read
    from disk: its pickle stream, and a list holding its only reference."""
    stream = pickle.dumps(Trie(), protocol=pickle.HIGHEST_PROTOCOL)
    holder = [pickle.loads(stream)]
    return memory.keep_read("k", holder, {"size": 100_000}, stream, 0.01)


def test_a_read_from_disk_kept_in_ram_goes_to_one_thread(monkeypatch):
    memory = InMemoryBackend()
    seen: list = []
    lock = _gap_after_keeping(monkeypatch, memory, lambda: seen.append(memory.get("k")[1]))

    first = _read_from_disk(memory)

    assert lock.ran == 1 and not lock.errors, lock.errors
    assert first.root == PLAIN and seen[0].root == PLAIN
    assert seen[0] is not first, "two threads were handed one object"


def test_a_read_from_disk_let_go_while_it_is_handed_out_still_reaches_its_caller(monkeypatch):
    memory = InMemoryBackend()
    lock = _gap_after_keeping(monkeypatch, memory, lambda: memory.delete("k"))

    first = _read_from_disk(memory)

    assert lock.ran == 1 and not lock.errors, lock.errors
    assert first is not None and first.root == PLAIN
